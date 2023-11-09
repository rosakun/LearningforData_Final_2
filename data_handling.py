"""Module for handling datasets.

This module contains functions for reading dataset files into a unified data
structure. The data structure is a nametuple with two lists: the first list
contains the documents and the second list contains the labels.

The datasets are standardized to correspond to the OLID labeling scheme, where
offesinve documents are labeled "OFF" and non-offensive documents are labeled "NOT".
Additionally, various preprocessing steps are performed on the datasets, such as
converting emojis to text format, replacing occurences of '@USER' occuring over
3 times in a row with @USER @USER @USER, removing occurences of 'URL', and
converting text to lowercase.

The module provides various helper functions for merging a list of datasets
into a single dataset, shuffling a dataset, splitting a dataset into train and
dev portions, calculating the balance between OFF and NOT labels in a dataset.
"""


import csv
import random
import re
from collections import namedtuple
from typing import Callable

import emoji
import datasets


OLID_PATH = "data/OLID"
DYNABENCH_PATH = "data/dynabench/raw.csv"
SENTIMENT_PATH = "data/sentiment/raw.csv"


Dataset = namedtuple("Dataset", ["documents", "labels"])
DataReader = Callable[[], Dataset]


def read_olid_data() -> dict[str, Dataset]:
    """Reads OLID train and dev files into a dictionary.

    Reads the OLID train and dev files, compliles corresponding Dataset objects
    and returns them in the form of a dictionary.

    Original dataset can be found at: https://scholar.harvard.edu/malmasi/olid.
    """

    documents_train, labels_train = [], []

    with open(f"{OLID_PATH}/train_preprocessed.tsv", encoding="UTF-8") as olid_train:
        reader = csv.reader(olid_train, delimiter="\t")
        for row in reader:
            documents_train.append(row[0])
            labels_train.append(row[1])

        train = Dataset(documents_train, labels_train)

    documents_dev, labels_dev = [], []

    with open(f"{OLID_PATH}/dev_preprocessed.tsv", encoding="UTF-8") as olid_dev:
        reader = csv.reader(olid_dev, delimiter="\t")
        for row in reader:
            documents_dev.append(row[0])
            labels_dev.append(row[1])

        dev = Dataset(documents_dev, labels_dev)

    return {"train": train, "dev": dev}


def read_berkeley_data() -> Dataset:
    """Reads Berkeley data and returns a list of documents and a list of labels.

    Reads the Berkeley Hate Speech dataset. Extracts text values as documents
    and converts hate_speech_score values greater than or equal to 0.5 to "OFF".

    Dataset can be found at:
    https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech.
    """

    dataset = datasets.load_dataset("ucberkeley-dlab/measuring-hate-speech", "binary")
    df = dataset["train"].to_pandas()
    documents = df["text"].tolist()
    scores = df["hate_speech_score"].tolist()
    labels = ["OFF" if score > 0.5 else "NOT" for score in scores]

    return Dataset(documents, labels)


def read_dynabench_data() -> Dataset:
    """Reads Dynabench data and returns a list of documents and a list of labels.

    Reads the Dynamically Generated Hate Speech Dataset. Extracts text values as
    documents and converts label "hate" to "OFF" and "nothate" as "NOT".

    Dataset can be found at:
    https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset
    """

    documents, labels = [], []

    with open(DYNABENCH_PATH, encoding="UTF-8") as dynabench_f:
        reader = csv.DictReader(dynabench_f)
        for row in reader:
            document = row["text"]
            label = "OFF" if row["label"] == "hate" else "NOT"
            documents.append(document)
            labels.append(label)

    return Dataset(documents, labels)


def read_sentiment_data() -> Dataset:
    """Reads Sentiment data and returns a list of documents and a list of labels.

    Reads Twitter Sentiment Analysis dataset. Extracts 'tweet' values as documents
    and converts 'label' values to "OFF" if they are 1 (i.e. racist/sexist),
    otherwise "NOT".

    Dataset can be found at:
    https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech.
    """

    documents, labels = [], []

    with open(SENTIMENT_PATH, encoding="UTF-8") as sentiment_f:
        reader = csv.DictReader(sentiment_f)
        for row in reader:
            document = row["tweet"].strip()
            label = "OFF" if row["label"] == "1" else "NOT"
            documents.append(document)
            labels.append(label)

    return Dataset(documents, labels)


def stats(dataset: Dataset) -> dict:
    """Calculates the balance between OFF and NOT labels in a dataset."""

    stats = {
        "total": len(dataset.labels),
        "total_off": dataset.labels.count("OFF"),
        "total_not": dataset.labels.count("NOT"),
        "fract_off": dataset.labels.count("OFF") / len(dataset.labels)
    }

    return stats


def preprocess(dataset: Dataset, demojize: bool = True, trunc_user: bool = True,
               replace_url: bool = True, lowercase: bool = True) -> None:
    """Performs various preprocessing steps on a dataset.

    Performs the following preprocessing steps on a dataset:
    - Convert emojis to text format
    - Replace occurences of '@USER' occuring over 3 times in a row with @USER @USER @USER
    - Remove occurences of 'URL'
    - Convert text to lowercase

    Steps can be turned on or off by setting the corresponding boolean argument.
    """

    for i, doc in enumerate(dataset.documents):
        if demojize: doc = emoji.demojize(doc)
        if trunc_user: doc = re.sub(r"(@USER\s+){2,}", "@USER @USER ", doc)
        if replace_url: doc = doc.replace("URL", "")
        if lowercase: doc = doc.casefold()
        dataset.documents[i] = doc


def merge(datasets: list[Dataset]) -> Dataset:
    """Merges a list of datasets into a single dataset."""

    documents, labels = [], []

    for dataset in datasets:
        documents.extend(dataset.documents)
        labels.extend(dataset.labels)

    return Dataset(documents, labels)


def shuffle(dataset: Dataset) -> Dataset:
    """Returnes a shuffled version of a dataset."""

    pairs = list(zip(dataset.documents, dataset.labels))
    random.shuffle(pairs)
    documents, labels = [], []
    for document, label in pairs:
        documents.append(document)
        labels.append(label)

    return Dataset(documents, labels)


def train_dev_split(dataset: Dataset, ratio: float = 0.8) -> dict[str, Dataset]:
    """Splits a dataset into train and dev portions based on a ratio."""

    split_idx = int(len(dataset.documents) * ratio)

    split = {
        "train": Dataset(dataset.documents[:split_idx], dataset.labels[:split_idx]),
        "dev": Dataset(dataset.documents[split_idx:], dataset.labels[split_idx:])
    }

    return split


def to_csv(dataset: Dataset, path: str, delimiter: str = ",") -> None:
    """Writes a dataset to a CSV file with a specified delimiter.

    Writes a dataset to a CSV file, where the first column contains the documents
    and the second column contains the labels.
    """

    with open(path, "w", encoding="UTF-8") as csv_f:
        writer = csv.writer(csv_f, delimiter=delimiter)
        for document, label in zip(dataset.documents, dataset.labels):
            writer.writerow([document, label])


def pipeline(dataset: Dataset, train_path: str, dev_path: str, to_preprocess=True) -> None:
    """Runs the pipeline on a dataset and writes it to a CSV file.

    Preprocesses the dataset, splits it into train and dev portions, and writes
    the portions to corresponding TSV files.
    """

    if to_preprocess: preprocess(dataset)
    split = train_dev_split(dataset)
    print("Train:", stats(split["train"]))
    print("Dev:", stats(split["dev"]))
    to_csv(split["train"], train_path)
    to_csv(split["dev"], dev_path)


def main() -> None:
    """Main function for testing and demonstrating purposes."""

    olid = read_olid_data()
    olid = merge(list(olid.values()))

    berkeley = read_berkeley_data()
    pipeline(berkeley, "data/berkeley/train.csv", "data/berkeley/dev.csv")

    dynabench = read_dynabench_data()
    pipeline(dynabench, "data/dynabench/train.csv", "data/dynabench/dev.csv")

    sentiment = read_sentiment_data()
    pipeline(sentiment, "data/sentiment/train.csv", "data/sentiment/dev.csv")

    merged = merge([olid, berkeley, dynabench, sentiment])
    merged = shuffle(merged)
    pipeline(merged, "data/merged/train.csv", "data/merged/dev.csv", to_preprocess=False)


if __name__ == "__main__":
    main()
