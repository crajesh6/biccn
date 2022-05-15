

# ==============================================================================
# Imports
# ==============================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

import logomaker
from biccn.data import WindowedGenomeDataset
from biccn.model_zoo import residualbind


# ==============================================================================
# Functions
# ==============================================================================
def saliency_maps(X, class_index, batch_size):

    return function_batch(X, class_index, saliency_map, batch_size)

def function_batch(X, class_index, fun, batch_size):

    dataset = tf.data.Dataset.from_tensor_slices(X)
    outputs = []

    for batch in dataset.batch(batch_size):
        outputs.append(fun(batch, class_index))

    return np.concatenate(outputs, axis=0)

def saliency_map(batch, class_index):

    if not tf.is_tensor(batch):
        batch = tf.Variable(batch)

    with tf.GradientTape() as tape:
        tape.watch(batch)
        outputs = model(batch)[:, class_index]

    return tape.gradient(outputs, batch)


# ==============================================================================
# Hyperparameters
# ==============================================================================



# ==============================================================================
# Load and prepare data (load data and create train, validation and test set)
# ==============================================================================
print("Loading data...")

AUTO = tf.data.experimental.AUTOTUNE
THRESHOLD = 0.0006867924257022952
BATCH_SIZE = 128

DATA_DIR = "data"

test_ds = WindowedGenomeDataset(
    target_path="data/targets.bed",
    fasta_path="data/GRCm38.fa",
    chromosomes=[f"chr{i}" for i in [1, 3, 5]],
    window_size=999,
    dtype=np.float32
)

X_test = test_ds[:]['inputs']
y_test = np.where(test_ds[:]['labels'] > THRESHOLD, 1.0, 0.0)

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
    rc=False
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0),
    metrics=[
        tf.keras.metrics.AUC(curve="ROC", name="auroc"),
        tf.keras.metrics.AUC(curve="PR", name="aupr")
    ]
)
model.load_weights('models/resbind_weights_2.0.h5')


print("Model creation complete.")


# ==============================================================================
# Saliency Maps
# ==============================================================================
print("Computing Saliency Maps...")

# plot saliency maps
attr_score = {}
alphabet = 'ACGT'
num_plots = 20
plot_size = 250
classes = [9, 21, 22, 23, 12, 13, 14, 15, 25, 29, 30, 31, 32]

for _, class_index in tqdm(enumerate(classes), total=len(classes)):

    # get high predicted sequences
    pos_index = np.where((y_test[:, class_index] == 1) & (np.sum(y_test,axis=1) == 1))[0]
    print(f"Num. positive sequences for class {class_index}: {len(pos_index)}")

    if(len(pos_index) < 1):
        continue

    predictions = model.predict(X_test[pos_index])
    plot_index = pos_index[np.argsort(predictions[:, class_index])[::-1]]
    X = X_test[plot_index[0:num_plots]]

    # get attribution scores
    attr_score[class_index] = saliency_maps(X, class_index, batch_size=BATCH_SIZE)
    attr_score[class_index] = attr_score[class_index] * X


    # plot attribution scores for sequences with top predictions
    N, L, A = attr_score[class_index].shape
    fig = plt.figure(figsize=(25, num_plots))

    for k in tqdm(range(N)):
        score = np.max(attr_score[class_index][k], axis=1)
        index = np.argmax(score)
        start = np.maximum(0, index - int(plot_size/2))
        end = start + plot_size
        if end > L:
            start = L - plot_size
            end = L
        plot_range = range(start,end)

        counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(plot_size)))
        for a in range(A):
            for l in range(plot_size):
                counts_df.iloc[l,a] = attr_score[class_index][k, plot_range[l], a]

        ax = plt.subplot(num_plots, 1, k+1)
        logomaker.Logo(counts_df, ax=ax)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])


    outfile = f"results/saliency/saliency_{class_index}__250_2.0.pdf"
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')

print("Saliency analysis complete.")
