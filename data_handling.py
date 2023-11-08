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

The module provides a function for merging a list of datasets into a single
dataset.

The module contains a function for splitting a dataset into train and dev portions.
The module also contains a function for calculating the balance between OFF and
NOT labels in a dataset.
"""


import csv
import re
from collections import namedtuple
from typing import Callable

import emoji
import datasets


JIGSAW_PATH = "data/jigsaw/raw.csv"
STORMFRONT_PATH = "data/stormfront"
DYNABENCH_PATH = "data/dynabench/raw.csv"
SENTIMENT_PATH = "data/sentiment/raw.csv"
ICWSM18_PATH = "data/ICWSM18/all.csv"  # NOT USED


Dataset = namedtuple("Dataset", ["documents", "labels"])
DataReader: Callable[[], Dataset]


def read_jigsaw_data() -> Dataset:
    """Reads Jigsaw data and returns a list of documents and a list of labels.

    Reads the Jigsaw Unintended Bias in Toxicity Classification dataset.
    Extracts comment_text values as documents and converts target values
    greater than or equal to 0.5 to "OFF" and less than 0.5 to "NOT" as labels.

    Dataset can be found at:
    https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data.
    """

    documents, labels = [], []

    with open(JIGSAW_PATH, encoding="UTF-8") as kaggle_f:
        reader = csv.DictReader(kaggle_f)
        for row in reader:
            document = row["comment_text"]
            label = "OFF" if float(row["target"]) >= 0.5 else "NOT"
            documents.append(document)
            labels.append(label)

    return Dataset(documents, labels)


def read_stormfront_data() -> Dataset:
    """Reads Stormfront data and returns a list of documents and a list of labels.

    Reads the Stormfront Hate Speech Corpus dataset. Label is assigned based on
    the value of the label column in annotations_metadata.csv. If the value is
    "hate", the label is "OFF", otherwise it is "NOT". Comment text is extracted
    from the .txt file correspinding to file_id value.

    Dataset can be found at: https://huggingface.co/datasets/hate_speech18.
    """

    documents, labels = [], []

    with open(f"{STORMFRONT_PATH}/annotations_metadata.csv", encoding="UTF-8") as stormfront_f:
        reader = csv.DictReader(stormfront_f)
        for row in reader:
            label = "OFF" if row["label"] == "hate" else "NOT"
            doc_path = f"{STORMFRONT_PATH}/texts/{row['file_id']}.txt"
            with open(doc_path, encoding="UTF-8") as doc_f:
                document = doc_f.read().strip()
            documents.append(document)
            labels.append(label)

    return Dataset(documents, labels)


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


# NOT USED
def read_icwsm18_data() -> Dataset:
    """Reads ICWSM18 data and returns a list of documents and a list of labels.

    Reads all comments from the ICWSM18 dataset, converted to CSV from XLSX.
    Documents are extracted from the 'message' column. Labels are assigned based
    on the value of the 'Class' column ("OFF" if "Hateful", otherwise "NOT").

    Dataset can be found at:
    https://www.dropbox.com/s/21wtzy9arc5skr8/ICWSM18%20-%20SALMINEN%20ET%20AL.xlsx?dl=0.
    """

    documents, labels = [], []

    with open(ICWSM18_PATH, encoding="UTF-8") as icwsm18_f:
        reader = csv.DictReader(icwsm18_f, delimiter=";")
        for row in reader:
            document = row["message"].strip()
            label = "OFF" if row["Class"] == "Hateful" else "NOT"
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


def train_dev_split(dataset: Dataset, ratio: float = 0.8) -> dict[str, Dataset]:
    """Splits a dataset into train and dev portions based on a ratio."""

    split_idx = int(len(dataset.documents) * ratio)

    split = {
        "train": Dataset(dataset.documents[:split_idx], dataset.labels[:split_idx]),
        "dev": Dataset(dataset.documents[split_idx:], dataset.labels[split_idx:])
    }

    return split


def to_tsv(dataset: Dataset, path: str) -> None:
    """Writes a dataset to a TSV file.

    Writes a dataset to a TSV file, where the first column contains the documents
    and the second column contains the labels.
    """

    with open(path, "w", encoding="UTF-8") as tsv_f:
        writer = csv.writer(tsv_f, delimiter="\t")
        for document, label in zip(dataset.documents, dataset.labels):
            writer.writerow([document, label])


def pipeline(dataset: Dataset, train_path: str, dev_path: str) -> None:
    """Runs the pipeline on a dataset and writes it to a TSV file.

    Preprocesses the dataset, splits it into train and dev portions, and writes
    the portions to corresponding TSV files.
    """

    preprocess(dataset)
    split = train_dev_split(dataset)
    to_tsv(split["train"], train_path)
    to_tsv(split["dev"], dev_path)


def main() -> None:
    """Main function for testing and demonstrating purposes."""

    jigsaw = read_jigsaw_data()
    pipeline(jigsaw, "data/jigsaw/train.tsv", "data/jigsaw/dev.tsv")

    stormfront = read_stormfront_data()
    pipeline(stormfront, "data/stormfront/train.tsv", "data/stormfront/dev.tsv")

    berkeley = read_berkeley_data()
    pipeline(berkeley, "data/berkeley/train.tsv", "data/berkeley/dev.tsv")

    dynabench = read_dynabench_data()
    pipeline(dynabench, "data/dynabench/train.tsv", "data/dynabench/dev.tsv")

    sentiment = read_sentiment_data()
    pipeline(sentiment, "data/sentiment/train.tsv", "data/sentiment/dev.tsv")

    merged = merge([jigsaw, stormfront, berkeley, dynabench, sentiment])
    pipeline(merged, "data/merged/train.tsv", "data/merged/dev.tsv")


if __name__ == "__main__":
    main()
