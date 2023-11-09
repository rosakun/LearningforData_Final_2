"""Module for training a random forest classifier on the OLID dataset.

Example usage:
>>> python3 random_forest.py -tf train.tsv -df dev.tsv -t test.tsv -bow
"""


import argparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='data/OLID/train_preprocessed.tsv', type=str,
                        help="Train file to learn from (data/OLID/train_preprocessed.tsv)")
    parser.add_argument("-df", "--dev_file", default='data/OLID/dev_preprocessed.tsv', type=str,
                        help="Dev file to evaluate on (default data/OLID/dev_preprocessed.tsv)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    parser.add_argument("-bow", "--bow", action="store_true",
                        help="Use the Count Vectorizer")
    parser.add_argument("-tfidf", "--tfidf", action="store_true",
                        help="Use the TF-IDF Vectorizer")
    parser.add_argument("-u", "--union", action="store_true",
                        help= "Use the Union Vectorizer")
    parser.add_argument("-bi", "--bigram", action="store_true",
                        help="Use bigram features")
    parser.add_argument("-tri", "--trigram", action="store_true",
                        help="Use trigram features")
    parser.add_argument("-four", "--fourgram", action="store_true",
                        help="Use four-gram features")
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


def identity(inp):
    """Dummy function that just returns the input."""
    return inp

def create_model() -> Pipeline:
    """Creates a pipeline with a vectorizer and a classifier."""

    args = create_arg_parser()

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.

    if args.bow:
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity)
        vec_name = "CountVectorizer"

    if args.tfidf:
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
        vec_name = "TF-IDF"

    if args.union:
        count = CountVectorizer(preprocessor=identity, tokenizer=identity)
        tf_idf = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
        vec_name = "UnionVectorizer"
        vec = FeatureUnion([("count", count), ("tf", tf_idf)])

    if args.bigram:
        # applying ngram on both vectorizers
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity,
                                ngram_range=(1,2), min_df=0.01)
        vec_name = "BigramCount"

    if args.trigram:
        # applying ngram on both vectorizers
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity,
                                ngram_range=(1,3), min_df=0.01)
        vec_name = "TrigramCount"

    if args.fourgram:
        # applying ngram on both vectorizers
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity,
                                ngram_range=(1,4), min_df=0.01)
        vec_name = "Four-gramCount"

    classifier_name = "Random Forest"
    classifier_model = RandomForestClassifier(random_state=42)

    classifier = Pipeline([("vec", vec), ("cls", classifier_model)])
    print(f"\n Vectorizer: {vec_name}, Classifier: {classifier_name} \n")

    return classifier


def main():

    args = create_arg_parser()
    X_train, Y_train = read_corpus(args.train_file)
    classifier = create_model()
    pred = cross_val_predict(classifier, X_train, Y_train, cv=5)

    print("Validation set:")
    print(classification_report(Y_train, pred, digits=3))
    conf_matrix = confusion_matrix(Y_train,pred)
    print(conf_matrix)

    if args.test_file:
        # Read in test set and vectorize
        X_test, Y_test = read_corpus(args.test_file)
        # Finally do the predictions
        pred_test = cross_val_predict(classifier, X_test, Y_test, cv=5)
        print("Test set:")
        print(classification_report(Y_test,pred_test,digits=3))
        conf_matrix_test = confusion_matrix(Y_test,pred_test)
        print(conf_matrix_test)


if __name__ == "__main__":
    main()