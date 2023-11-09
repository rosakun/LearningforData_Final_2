"""Module for training and evaluating the LSTM model.

Hyperparameters are to be specified in the code. Optimal values already set.

Example usage:
>>> python3 lstm.py -i train.txt -d dev.txt -t test.txt
"""


import json
import argparse
import random as py_random

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, Activation
from keras.initializers import Constant
from sklearn.metrics import accuracy_score, recall_score, precision_score, precision_recall_curve, f1_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


# Random seed for reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)
py_random.seed(1234)


def create_arg_parser() -> argparse.Namespace:
    """Converts command line arguments to Namespace object and returns it."""

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--train_file", default="data/OLID/train_preprocessed.tsv", type=str,
                        help="Input file to learn from (default data/OLID/train_preprocessed.tsv)")
    parser.add_argument("-d", "--dev_file", type=str, default="data/OLID/train_preprocessed.tsv",
                        help="Separate dev set to read in (default data/OLID/train_preprocessed.tsv)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    parser.add_argument("-e", "--embeddings", default="glove_embeddings/glove_50d.json", type=str,
                        help="Embedding file we are using (default glove_reviews.json)")
    args = parser.parse_args()

    return args


def read_corpus(corpus_file: str) -> tuple[list[str], list[str]]:
    """Reads review data set and returns docs and labels."""

    documents, labels = [], []

    with open(corpus_file, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split()
            label = tokens[-1]
            tweet = ' '.join(tokens[:-1])
            documents.append(tweet)
            labels.append(label)
    return documents, labels


def read_embeddings(embeddings_file: str):
    """Reads in word embeddings from file and save as numpy array."""

    embeddings = json.load(open(embeddings_file, 'r', encoding='utf-8'))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc, emb):
    """Creates embedding matrix given vocab and the embeddings."""

    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    embedding_dim = len(emb["the"])
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def create_model(Y_train: list[str], emb_matrix) -> Sequential:
    """Creates the bidirectional LSTM model."""

    args = create_arg_parser()

    # Hyperparameters
    HIDDEN_UNITS = 64
    LEARNING_RATE = 0.01
    LOSS_FUNCTION = "binary_crossentropy"
    OPTIM = Adam(learning_rate=LEARNING_RATE)

    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])  # 300
    print(f"Embedding dim: {embedding_dim}")  # For debugging

    num_tokens = len(emb_matrix)
    print(f"Num tokens: {num_tokens}")  # For debugging

    num_labels = len(set(Y_train))
    print(f"Num labels: {num_labels}")  # For debugging

    model = Sequential()

    model.add(
        Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=False)        )

    model.add(
        LSTM(units=HIDDEN_UNITS, return_sequences=False, recurrent_dropout=0.5))

    model.add(Dropout(0.8))

    model.add(Dense(input_dim=10, units=1, activation='sigmoid'))

    model.compile(
        loss=LOSS_FUNCTION,
        optimizer=OPTIM,
        metrics=[
            tf.keras.metrics.Recall(
                thresholds=None,
                top_k=None,
                class_id=None,
                name=None,
                dtype=None
            )
        ]
    )

    return model


def train_model(model: Sequential,
                X_train: list[str], Y_train: list[str],
                X_dev: list[str], Y_dev: list[str]) -> Sequential:
    """Trains the LSTM model."""

    VERBOSE = 1
    BATCH_SIZE = 25
    EPOCHS = 20

    callback = EarlyStopping(monitor="val_loss", patience=3)
    model.fit(
        X_train,
        Y_train,
        verbose=VERBOSE,
        epochs=EPOCHS,
        callbacks=[callback],
        batch_size=BATCH_SIZE,
        validation_data=(X_dev, Y_dev)
    )
    test_set_predict(model, X_dev, Y_dev, "dev")

    return model

def custom_argmax(value, threshold=0.5):
    return 1 if value >= threshold else 0


def test_set_predict(model: Sequential, X_test: list[str], Y_test: list[str], ident: str) -> None:
    """Measure accuracy on own test set, which is a subset of the training data."""

    Y_pred = [custom_argmax(value) for value in model.predict(X_test)]

    formatted_acc = accuracy_score(Y_test, Y_pred)
    print(f"Accuracy on own {ident} set: {formatted_acc}")

    macro_prec = round(precision_score(Y_test, Y_pred, average='macro'), 3)
    macro_rec = round(recall_score(Y_test, Y_pred, average='macro'), 3)
    macro_f1 = round(f1_score(Y_test, Y_pred, average='macro'), 3)

    print(f"Macro-average on own {ident} set (prec-rec-f1): {macro_prec} {macro_rec} {macro_f1}")

    weighted_prec = round(precision_score(Y_test, Y_pred, average='weighted'), 3)
    weighted_rec = round(recall_score(Y_test, Y_pred, average='weighted'), 3)
    weighted_f1 = round(f1_score(Y_test, Y_pred, average='weighted'), 3)

    print(f"Weighted-average on own {ident} set (prec-rec-f1): {weighted_prec} {weighted_rec} {weighted_f1}")


def main() -> None:
    """Main function to train and test neural network given cmd line arguments."""

    args = create_arg_parser()

    # Read in the data and embeddings
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    embeddings = read_embeddings(args.embeddings)

    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)

    # Create model
    model = create_model(Y_train, emb_matrix)

    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    # Train the model
    model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin)

    # Do predictions on specified test set
    if args.test_file:
        # Read in test set and vectorize

        X_test, Y_test = read_corpus(args.test_file)

        Y_test_bin = encoder.fit_transform(Y_test)
        X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
        # Finally do the predictions
        test_set_predict(model, X_test_vect, Y_test_bin, "test")


if __name__ == "__main__":
    main()
