"""Module for reading and preprocessing data."""


import csv
from typing import Callable

import datasets


JIGSAW_PATH = "data/jigsaw/train.csv"
STORMFRONT_PATH = "data/stormfront"
DYNABENCH_PATH = "data/dynabench/data.csv"
SENTIMENT_PATH = "data/sentiment/train.csv"
ICWSM18_PATH = "data/ICWSM18/all.csv"  # NOT USED


Dataset: tuple[list[str], list[str]]  # List of documents and list of labels
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

    return documents, labels


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

    return documents, labels


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

    return documents, labels


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

    return documents, labels


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
            document = row["tweet"]
            label = "OFF" if row["label"] == "1" else "NOT"
            documents.append(document)
            labels.append(label)

    return documents, labels


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

    return documents, labels


def stats(dataset: Dataset) -> dict:
    """Calculates the balance between OFF and NOT labels in a dataset."""

    stats = {
        "total": len(dataset[1]),
        "total_off": dataset[1].count("OFF"),
        "total_not": dataset[1].count("NOT"),
        "fract_off": dataset[1].count("OFF") / len(dataset[1])
    }

    return stats


def main() -> None:
    """Main function for testing and demonstrating purposes."""

    datasets = [
        read_jigsaw_data(),
        read_stormfront_data(),
        read_berkeley_data(),
        read_dynabench_data(),
        read_sentiment_data()
    ]

    for dataset in datasets:
        print(stats(dataset))


if __name__ == "__main__":
    main()