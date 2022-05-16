# ==============================================================================
# Imports
# ==============================================================================
from collections import OrderedDict
import h5py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from pathlib import Path
import sys

import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import logomaker
import tfomics
from tfomics import explain

from biccn.data import WindowedGenomeDataset
from biccn.model_zoo import residualbind

# ==============================================================================
# Paths/Constants
# ==============================================================================
DATA_DIR = "./biccn/data"
MODEL_DIR = "./biccn/models"
SAVE_DIR = "./biccn/results"
AUTO = tf.data.experimental.AUTOTUNE
THRESHOLD = 0.01
BATCH_SIZE = 128
num_labels = 33


@click.command()
@click.argument("trial_name", type=str)
@click.argument("marker", type=str)
@click.argument("model_weights", type=str)
@click.argument("rc", type=bool)
def main(trial_name: str, marker: str, model_weights: str, rc: bool):

    TRIAL_PATH = os.path.join(SAVE_DIR, trial_name)
    Path(TRIAL_PATH).mkdir(exist_ok=True)

    SAVE_PATH = os.path.join(TRIAL_PATH, marker)
    Path(SAVE_PATH).mkdir(exist_ok=True)


    # ==============================================================================
    # Create Model
    # ==============================================================================
    print("Creating model...")
    model = residualbind(
        input_shape=(999, 4),
        output_shape=33,
        activation="exponential",
        num_units=[128, 256, 512, 512],
        rc = rc
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0),
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", name="auroc"),
            tf.keras.metrics.AUC(curve="PR", name="aupr")
        ]
    )


    weights_path = os.path.join(MODEL_DIR, model_weights)
    model.load_weights(weights_path)
    print("Model creation complete!")
    # ==============================================================================
    # Marker to Annotation Lookup Table
    # ==============================================================================
    annotation = '/mnt/1a18a49e-9a31-4dbf-accd-3fb8abbfab2d/brain_atac/data/marker_vs_cluster/BICCN_sub_cluster_and_annotation_file.csv'
    marker_cluster = pd.read_csv(annotation)
    major_types = marker_cluster.celltype.values
    sub_types = marker_cluster.annotation_file.values.astype(str)

    markers = {}
    for t,p in zip(["IN", "EX"], ["GABAergic_markers_fc_5k.bed", "Glutamatergic_markers_fc_5k.bed"]):

        markers[t] = {
            "bed_path": p,
            "classes": [i for i in range(len(major_types)) if t in major_types[i]]
         }

    for s in np.unique(sub_types):
        path = s.split("_location.bed")[0]
        bed_path = path + "_5k.bed"
        m = path.split("_")[-1]
        markers[m.upper()] = {
            "bed_path": bed_path,
            "classes": [i for i in range(len(sub_types)) if m in sub_types[i]]
         }


    # marker = sys.argv[1]
    marker_bed_path = f"data/markers_5k/{markers[marker]['bed_path']}"
    classes = markers[marker]["classes"]
    task = f"task{classes[0]}"
    tile_idx = len(classes)

    # ==============================================================================
    # Load and prepare data (cluster set)
    # ==============================================================================
    print("Loading data and computing saliency scores...")
    marker_ds = WindowedGenomeDataset(
        target_path=marker_bed_path,
        fasta_path=DATA_DIR + "/GRCm38.fa",
        window_size=4995,
        dtype=np.float32
    )

    X_test = marker_ds[:]['inputs']
    X_test = X_test.reshape(np.prod([X_test.shape[0], 5]), 999, 4)
    print(X_test.shape)

    # ==============================================================================
    # Compute Class Saliency Scores
    # ==============================================================================
    scores = OrderedDict()
    hyp_scores = OrderedDict()

    for class_index in classes:

        print(f"Class index: {class_index}\nNumber of positive sequences: {len(X_test)}")
        dataset = tf.data.Dataset.from_tensor_slices(X_test)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
        total = np.ceil(X_test.shape[0] / BATCH_SIZE)

        saliency = []

        for i, x in tqdm(enumerate(dataset), total=total):
          x = tf.Variable(x)

          with tf.GradientTape() as tape:
              tape.watch(x)
              outputs = model(x)[:, class_index]

          saliency += [tape.gradient(outputs, x)]

        saliency = np.concatenate(saliency, axis=0)

        scores[f"task{class_index}"] = saliency*X_test
        hyp_scores[f"task{class_index}"] = saliency - np.mean(saliency, axis=-1)[:,:,None]

    # ==============================================================================
    # Combine Cluster Scores and Sequences
    # ==============================================================================
    task_to_scores = OrderedDict()
    task_to_hyp_scores = OrderedDict()

    task_to_scores[task] = np.concatenate([val for (key, val) in scores.items()])
    task_to_hyp_scores[task] = np.concatenate([val for (key, val) in hyp_scores.items()])
    onehot_data = np.tile(X_test, (tile_idx,1,1))

    print(f"Length of scores: {len(task_to_hyp_scores[task])}")
    print(f"Length of data: {len(onehot_data)}")
    print("Finished computing saliency scores!")
    # ==============================================================================
    # Perform Modisco Analysis
    # ==============================================================================
    import time
    from importlib import reload
    import modisco
    import modisco.backend
    import modisco.nearest_neighbors
    import modisco.affinitymat
    import modisco.tfmodisco_workflow.seqlets_to_patterns
    import modisco.tfmodisco_workflow.workflow
    import modisco.aggregator
    import modisco.cluster
    import modisco.value_provider
    import modisco.core
    import modisco.coordproducers
    import modisco.metaclusterers
    import modisco.visualization
    from modisco.visualization import viz_sequence

    start_time = time.time()

    print("Running TF-MoDISco for task", task)

    tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(

                    sliding_window_size=15,
                    flank_size=5,
                    target_seqlet_fdr=0.15,
                    seqlets_to_patterns_factory=modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                        trim_to_window_size=15,
                        initial_flank_to_add=5,
                        kmer_len=5,
                        num_gaps=1,
                        num_mismatches=0,
                        final_min_cluster_size=30
                    )
                )(
                task_names=[task],
                contrib_scores=task_to_scores,
                hypothetical_contribs=task_to_hyp_scores,
                one_hot=onehot_data
                )

    print("Starting Metaclustering!")
    mc_idx = len(tfmodisco_results.metacluster_idx_to_submetacluster_results)

    for metacluster_idx in range(mc_idx):

        print(f"Metacluster {metacluster_idx} for task {task}")

        patterns = (tfmodisco_results.metacluster_idx_to_submetacluster_results[
            metacluster_idx
        ].seqlets_to_patterns_result.patterns)

        for pattern_idx, pattern in enumerate(patterns):

            print(f"Pattern idx {pattern_idx} with {len(pattern.seqlets)} seqlets")
            fig = plt.figure(figsize=(20, 2))
            ax = plt.subplot(111)
            viz_sequence.plot_weights_given_ax(
                ax=ax,
                array=pattern[f"{task}_contrib_scores"].fwd,
                height_padding_factor=0.2,
                length_padding=1.0,
                subticks_frequency=1.0,
                highlight={},
            )


            outfile = os.path.join(SAVE_PATH, f"modisco_class_{marker}_{metacluster_idx}_{pattern_idx}_{len(pattern.seqlets)}_5k.pdf")
            print(f"Saving figure at {outfile}")
            fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')

    # ==============================================================================
    # Save Results
    # ==============================================================================
    os.rename('figures/scoredist_0.png', f'figures/scoredist_{trial_name}_{marker}.png')
    results_file = os.path.join(SAVE_PATH, f"modisco_{marker}_results_5k.hdf5")
    # tfmodisco_results.save_hdf5(h5py.File(f"{SAVE_DIR}modisco_{marker}_results_5k.hdf5"))
    tfmodisco_results.save_hdf5(h5py.File(results_file))
    del tfmodisco_results

    end_time = time.time()
    print(f"Finished modisco analysis!\nTook {(end_time-start_time)} seconds.")

    # ==============================================================================
    # Plot attribution scores for sequences with top predictions
    # ==============================================================================
    print(f"Plotting attribution scores...")
    idx = np.random.randint(0, len(onehot_data), 50)
    attr_scores = task_to_scores[task][idx]

    plot_size = 250
    num_plots = 50
    alphabet = 'ACGT'
    N, L, A = attr_scores.shape

    fig = plt.figure(figsize=(25, num_plots))

    for k in tqdm(range(N)):

        score = np.max(attr_scores[k], axis=1)
        index = np.argmax(score)
        start = np.maximum(0, index-int(plot_size/2))
        end = start + plot_size

        if end > L:
            start = L - plot_size
            end = L
        plot_range = range(start,end)

        counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(plot_size)))
        for a in range(A):
            for l in range(plot_size):
                counts_df.iloc[l, a] = attr_scores[k, plot_range[l], a]

        ax = plt.subplot(num_plots, 1, k+1)
        logomaker.Logo(counts_df, ax=ax)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])

    # outfile = f"{SAVE_DIR}{marker}_examples_5k.pdf"
    outfile = os.path.join(SAVE_PATH, f"{marker}_examples_5k.pdf")
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')


    # ==============================================================================
    # Save attribution scores to h5 file
    # ==============================================================================
    print("Saving Selected Attribution Scores...")

    scores_file = os.path.join(SAVE_PATH, f"{marker}_sampled_scores.h5")
    if (os.path.isfile(scores_file)):
        cmd = 'rm ' + scores_file
        subprocess.call(cmd, shell=True)

    with h5py.File(scores_file, "a") as f:

        f.create_dataset(
          f"scores",
          data=attr_scores,
          compression='gzip'
        )
    print("Done!")

if __name__ == "__main__":
    main()
