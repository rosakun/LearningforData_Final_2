"""Module for reading and preprocessing data."""


import csv
from typing import Callable

import datasets


JIGSAW_PATH = "data/jigsaw/train.csv"
STORMFRONT_PATH = "data/stormfront"
DYNABENCH_PATH = "data/dynabench/data.csv"


DataReader: Callable[[], tuple[list[str], list[str]]]


def read_jigsaw_data() -> tuple[list[str], list[str]]:
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


def read_stormfront_data() -> tuple[list[str], list[str]]:
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


def read_berkeley_data() -> tuple[list[str], list[str]]:
    """Reads Berkeley data and returns a list of documents and a list of labels.

    Reads the Berkeley Hate Speech dataset. Extracts text values as documents
    and converts hate_speech_score values greater than or equal to 0.5 to "OFF".

    Dataset can be found at:
    https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech.
    """

    dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')
    df = dataset["train"].to_pandas()
    documents = df["text"].tolist()
    scores = df["hate_speech_score"].tolist()
    labels = ["OFF" if score >= 0.5 else "NOT" for score in scores]

    return documents, labels


def read_dynabench_data() -> tuple[list[str], list[str]]:
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


def main():
    """Main function for testing."""

    documents, labels = read_berkeley_data()
    print(documents[:10], labels[:10])


if __name__ == "__main__":
    main()