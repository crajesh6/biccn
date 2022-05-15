import os
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Callable, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


def one_hot_encode(
        sequence: str,
        alphabet: str = "ACGT",
        neutral_alphabet: str = "N",
        neutral_value: Any = 0,
        dtype=np.float32
    ) -> np.ndarray:
    """One-hot encode sequence."""

    def to_uint8(s):
        return np.frombuffer(s.encode("ascii"), dtype=np.uint8)

    lookup = np.zeros([np.iinfo(np.uint8).max, len(alphabet)], dtype=dtype)
    lookup[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    lookup[to_uint8(neutral_alphabet)] = neutral_value
    lookup = lookup.astype(dtype)
    return lookup[to_uint8(sequence)]


class WindowedGenomeDataset:

    def __init__(
            self,
            target_path: str,
            fasta_path: str,
            window_size: int = None,
            chromosomes: List[str] = None,
            dtype: np.float32 = None
        ):
        self.target_path = target_path
        self.fasta_path = fasta_path
        self.chromosomes = chromosomes
        self.dtype = dtype

        df = pd.read_table(target_path, header=None, sep="\t")
        c = df[0].isin(chromosomes) if chromosomes else np.array(len(df)*[True])
        self.windows = df.loc[c, :2].values
        self.targets = df.loc[c, 3:].values.astype(self.dtype)

        _, start, stop = self.windows[0]
        self.window_size = window_size or (stop - start)

        self.input_shape = (self.window_size, 4)
        self.output_shape = (self.targets.shape[1],)

        self.genome = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item: Union[int, slice]):
        if not self.genome:
            from pyfaidx import Fasta
            self.genome = Fasta(self.fasta_path)

        chrom = self.windows[item, 0]
        start = self.windows[item, 1]
        stop = self.windows[item, 2]

        labels = self.targets[item]

        if isinstance(item, slice):
            seq = ""
            for chr, i, j in zip(chrom, start, stop):
                seq += self.genome[chr][i:j].seq.upper()

            inputs = one_hot_encode(seq)
            inputs = inputs.reshape(len(chrom), self.window_size, 4)

            item = range(item.start or 0, item.stop or len(self), item.step or 1)
            item = np.array(list(item), dtype=np.int32)
        else:
            seq = self.genome[chrom][start:stop].seq.upper()
            inputs = one_hot_encode(seq)

            item = item if item >= 0 else len(self) + item

        output = {
            "inputs": inputs,
            "labels": labels,
            "meta": {
                "id": item,
                "chrom": chrom,
                "start": start,
                "stop": stop
            }
        }
        return output


    def to_tfrecord(self, path: str, indices: List[int] = None):

        def serialize_example(id, inputs, labels, chrom, start, stop):

            inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
            inputs = tf.io.serialize_tensor(inputs).numpy()
            labels = tf.convert_to_tensor(labels, dtype=tf.float32)
            labels = tf.io.serialize_tensor(labels).numpy()

            chrom = chrom.encode("utf-8")

            feature = {
                "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[id])),
                "inputs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[inputs])),
                "labels": tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels])),
                "chrom": tf.train.Feature(bytes_list=tf.train.BytesList(value=[chrom])),
                "start": tf.train.Feature(int64_list=tf.train.Int64List(value=[start])),
                "stop": tf.train.Feature(int64_list=tf.train.Int64List(value=[stop]))
            }

            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()

        def generator():
            for i in tqdm(indices or range(len(self))):
                output = self[i]
                yield serialize_example(
                    id=output["meta"]["id"],
                    inputs=output["inputs"],
                    labels=output["labels"],
                    chrom=output["meta"]["chrom"],
                    start=output["meta"]["start"],
                    stop=output["meta"]["stop"]
                )

        serialized_features_dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=tf.string,
            output_shapes=()
        )

        writer = tf.data.experimental.TFRecordWriter(path)
        writer.write(serialized_features_dataset)
