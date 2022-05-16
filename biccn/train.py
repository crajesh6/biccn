

# ==============================================================================
# Insert imports here
# ==============================================================================
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import click
import numpy as np
import pandas as pd
import tensorflow as tf

from biccn.data import WindowedGenomeDataset
from biccn.model_zoo import residualbind

# ==============================================================================
# Paths
# ==============================================================================
DATA_DIR = "./biccn/data"
SAVE_DIR = "./biccn/models"


# ==============================================================================
# Hyperparameters
# ==============================================================================
AUTO = tf.data.experimental.AUTOTUNE
THRESHOLD = 0.0006867924257022952
BATCH_SIZE = 128


# ==============================================================================
# Load and prepare data (load data and create train, validation and test set)
# ==============================================================================
print("Loading data...")

def parse_tfrecord(example):
    feature_description = {
        "id": tf.io.FixedLenFeature([], tf.int64),
        "inputs": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.FixedLenFeature([], tf.string),
        "chrom": tf.io.FixedLenFeature([], tf.string),
        "start": tf.io.FixedLenFeature([], tf.int64),
        "stop": tf.io.FixedLenFeature([], tf.int64)
    }

    record = tf.io.parse_single_example(example, feature_description)
    record["inputs"] = tf.io.parse_tensor(record["inputs"], out_type=tf.float32)
    record["labels"] = tf.io.parse_tensor(record["labels"], out_type=tf.float32)

    inputs = tf.reshape(record["inputs"] , [999, 4])
    labels = tf.reshape(record["labels"], [33,])

    return inputs, labels

def load_dataset(filenames):

    records = tf.data.TFRecordDataset(filenames, num_parallel_reads=4)
    records = records.map(parse_tfrecord, num_parallel_calls=4)

    def data_augment(inputs, labels):
        return inputs, tf.where(labels > THRESHOLD, 1.0, 0.0)

    return records.map(data_augment, num_parallel_calls=4)

def get_dataset(batch_size: int, name: str):

    if name == "train":
        dataset = load_dataset([DATA_DIR + "/train.tfrec"])
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat()
    elif name == "valid":
        dataset = load_dataset([DATA_DIR + "/valid.tfrec"])
    elif name == "test":
        dataset = load_dataset([DATA_DIR + "/test.tfrec"])

    dataset = dataset.batch(batch_size).prefetch(AUTO)

    return dataset

@click.command()
@click.argument("trial_name", type=str)
def main(trial_name: str):

    SAVE_PATH = os.path.join(SAVE_DIR, trial_name)
    Path(SAVE_PATH).mkdir(exist_ok=True)
    items = [
        ("train", [2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, "X"]),
        ("valid", [7, 9]),
        ("test", [1, 3, 5])
    ]

    save = True # set to False if TFrecords already made

    datasets = {}

    for name, chroms in items:
        datasets[name] = WindowedGenomeDataset(
            target_path=DATA_DIR + "/targets.bed",
            fasta_path=DATA_DIR + "/GRCm38.fa",
            chromosomes=[f"chr{i}" for i in chroms],
            window_size=999,
            dtype=np.float32
        )
        if save:
            datasets[name].to_tfrecord(f"{DATA_DIR}/{name}.tfrec")

    train_size = len(datasets['train'])
    valid_size = len(datasets['valid'])
    test_size = len(datasets['test'])
    print("Train: ", train_size)
    print("Validation: ", valid_size)
    print("Validation: ", test_size)

    train_ds = get_dataset(BATCH_SIZE, name="train")
    valid_ds = get_dataset(BATCH_SIZE, name="valid")
    test_ds = get_dataset(BATCH_SIZE, name="test")

    print("Data loading complete.")

    # ==============================================================================
    # Model (build model architecture)
    # ==============================================================================
    print("Creating model...")

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

    print("Model creation complete.")


    # ==============================================================================
    # Train
    # ==============================================================================
    print("Training...")

    train_steps = int(np.ceil(train_size / BATCH_SIZE))
    valid_steps = int(np.ceil(valid_size / BATCH_SIZE))

    history = model.fit(
        train_ds,
        epochs=25,
        steps_per_epoch=train_steps,
        validation_data=valid_ds,
        validation_steps=valid_steps,
        callbacks=[
             tf.keras.callbacks.EarlyStopping(
                monitor="val_aupr",
                patience=10,
                verbose=1,
                mode="max",
                restore_best_weights=True
            ),
             tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_aupr",
                factor=0.2,
                patience=3,
                min_lr=1e-7,
                mode="max",
                verbose=1
            )
        ]
    )

    # Save model
    weights_path = os.path.join(SAVE_PATH, f"{trial_name}_weights.h5")
    model.save_weights(weights_path)

    history_path = os.path.join(SAVE_PATH, f"{trial_name}_history.csv")
    pd.DataFrame(history.history).to_csv(history_path)
    print("Training complete.")


    # ==============================================================================
    # Evaluate
    # ==============================================================================
    print("Evaluating...")
    test_steps = int(np.ceil(test_size / BATCH_SIZE))
    # test_steps = 100
    results = model.evaluate(test_ds, steps=test_steps)

    eval_path = os.path.join(SAVE_PATH, f"{trial_name}_eval.csv")
    pd.DataFrame(results).to_csv(eval_path)

    print("Evaluation complete.")


    # ==============================================================================
    # Results
    # ==============================================================================

    # Save the recorded performance in csv file or however you wish to use it.


    print(f"Done training model {trial_name}!")

if __name__ == "__main__":
    main()
