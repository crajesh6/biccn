

# ==============================================================================
# Imports
# ==============================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import click
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from biccn.data import WindowedGenomeDataset
from biccn.model_zoo import residualbind, remainder

# ==============================================================================
# Paths
# ==============================================================================

DATA_DIR = "/home/chandana/projects/biccn/data"
SAVE_DIR = "/home/chandana/projects/biccn/results"
MODEL_DIR = "/home/chandana/projects/biccn/models"


AUTO = tf.data.experimental.AUTOTUNE
THRESHOLD = 0.0006867924257022952
BATCH_SIZE = 128
test_size = 482812
total = np.ceil(test_size / BATCH_SIZE)
classes = list(range(33))
# classes = [9, 22]

num_filters = 128
act_thrshld = 0.5

# ==============================================================================
# Functions
# ==============================================================================
def clip_filters(W, threshold=0.5, pad=3):

    W_clipped = []
    for w in W:
        L,A = w.shape
        entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=1)
        index = np.where(entropy > threshold)[0]
        if index.any():
            start = np.maximum(np.min(index)-pad, 0)
            end = np.minimum(np.max(index)+pad+1, L)
            W_clipped.append(w[start:end,:])
        else:
            W_clipped.append(w)

    return W_clipped

def meme_generate(W, output_file='meme.txt', prefix='filter'):

    # background frequency
    nt_freqs = [1./4 for i in range(4)]

    # open file for writing
    f = open(output_file, 'w')

    # print intro material
    f.write('MEME version 4\n')
    f.write('\n')
    f.write('ALPHABET= ACGT\n')
    f.write('\n')
    f.write('Background letter frequencies:\n')
    f.write('A %.4f C  %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
    f.write('\n')

    for j, pwm in enumerate(W):
        L, A = pwm.shape
        f.write('MOTIF %s%d \n' % (prefix, j))
        f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (L, L))
        for i in range(L):
            f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[i,:]))
        f.write('\n')

    f.close()

@click.command()
@click.argument("model_name", type=str)
@click.argument("model_weights", type=str)
@click.argument("rc", type=bool)
def main(model_name: str, model_weights: str, rc: bool):

    print("Loading data...")

    test_ds = WindowedGenomeDataset(
        target_path=f"{DATA_DIR}/targets.bed",
        fasta_path=f"{DATA_DIR}/GRCm38.fa",
        chromosomes=[f"chr{i}" for i in [1, 3, 5]],
        window_size=999,
        dtype=np.float32
    )

    X_test = tf.data.Dataset.from_tensor_slices(test_ds[:]['inputs'])
    subset_ds = X_test.batch(BATCH_SIZE).prefetch(AUTO)

    print("Data loading complete.")

    # ==============================================================================
    # Model (build model architecture)
    # ==============================================================================
    print("Creating models...")

    model = residualbind(
        input_shape=(999, 4),
        output_shape=33,
        activation="exponential",
        num_units=[128, 256, 512, 512],
        rc=rc
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


    intermediate = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[3].output
    )

    print("Model creation complete.")


    # ==============================================================================
    # Filter Importance
    # ==============================================================================
    print("Computing Filter Importance...")


    s = WindowedGenomeDataset(
    target_path="targets.bed",
    fasta_path="GRCm38.fa",
    chromosomes=[f"chr{i}" for i in [1, 3, 5]],
    window_size=999,
    dtype=np.float32
    )

    AUTO = tf.data.experimental.AUTOTUNE
    THRESHOLD = 0.0006867924257022952
    BATCH_SIZE = 128
    test_size = 482812 # s[:]['labels'].shape[0]
    stride = BATCH_SIZE ** 2
    iterations = round(test_size / stride)

    outfile = os.path.join(SAVE_DIR, f"{model_name}_filter_influence.csv")
    pd.DataFrame(data).to_csv(outfile, header=True, index=False)

    print("Filter importance complete!")

if __name__ == "__main__":
    main()
