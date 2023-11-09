# Learning from Data Final Project

This project is concerned with the task of speech classification into two classes: `OFF` (offensive) and `NOT` (not offensive). To this end, we employ several models:

- Random Forest and LSTM are used as baselines
- Four BERT models each fine-tuned on an individual dataset (OLID, Berkeley, Dynabench, and Sentiment)
- A BERT modes fine-tuned on one big aggregated dataset containing entries from the above datasets

## Data

Datasets can be found [here](https://www.dropbox.com/scl/fo/5a7szayu80nkpuzb029mt/h?rlkey=v2jgv6ljz3d9fwe4hmz5c435b&dl=0).

In order for the script to work, place the `data` folder in the root directory with the code.

## Random Forest Classifier

### Dependencies

The following dependencies are required to use the Random Forest Classifier module:

- [sklearn](https://scikit-learn.org/stable/) - Version 1.3.2

### Usage

Run the random_forest.py file using the following command:

```
python random_forest.py
```
This command will use the default training and development files, as well as print out training and validation performance metrics.
The default training and development files are data/OLID/train_preprocessed.tsv and data/OLID/dev_preprocessed.tsv, respectively.

#### Testing

If you want to train the model and evaluate it on a test set, you can use the -t argument followed by the path to your test set file. For example:

```
python random_forest.py -t data/my_test_data.tsv
```
By specifying the test set, the code will train the model on the training data, evaluate it on the development data, and finally test it on the provided test set.

#### Feature Specification

This section will guide you on how to use feature specification in your code. You have the option to specify what features are used for input vectors by providing terminal arguments. The available options for feature specification are TF-IDF, UnionVector, bigrams, trigrams, or fourgrams. By default, the code uses a Bag of Words (BOW) count vectorizer.

To specify the features for your input vectors, you can use the following terminal arguments when running your code:

TF-IDF: Use `-tfidf` to use TF-IDF (Term Frequency-Inverse Document Frequency) vectorization for feature extraction.
UnionVector: Use `-u` to use the UnionVector, i.e. a concatenation of BOW and TF-IDF vectorization for feature extraction.
Bigrams: Use `-bi` to use bigrams for feature extraction.
Trigrams: Use `-tri` to use trigrams for feature extraction.
Fourgrams: Use `-four` to use fourgrams for feature extraction.

Example code:

```
python random_forest.py -four -t data/my_test_data.tsv
```
This code runs the Random Forest Classifier module with a fourgram vectorizer and evaluates it on the test set.

## LSTM

### Dependencies

The following dependencies are required to use the LSTM module:

- [numpy](https://numpy.org) - Version 1.26.1
- [tensorflow](https://www.tensorflow.org/install) - Version 2.13.0
- [keras](https://keras.io/getting_started/) - Version 2.13.1
- [sklearn](https://scikit-learn.org/stable/) - Version 1.3.2

### Usage

Run the lstm.py file using the following command:

```
python lstm.py
```

This command will use the default training and development files, as well as print out training and validation performance metrics.
The default training and development files are data/train_clean.tsv and data/dev_clean.tsv, respectively.

#### Testing

If you want to train the model and evaluate it on a test set, you can use the -t argument followed by the path to your test set file. For example:

```
python lstm.py -t data/my_test_data.tsv
```
By specifying the test set, the code will train the model on the training data, evaluate it on the development data, and finally test it on the provided test set.

#### Other Arguments

You can specify the input training and development files using the following optional command-line arguments:

* -i or --input: To specify the training file, use the -i argument followed by the path to your training file. For example:

```
python lstm.py -i data/my_training_data.tsv
```

* -d or --dev: To specify the development file, use the -d argument followed by the path to your development file. For example:
```
python lstm.py -d data/my_dev_data.tsv
```

* -e or --embeddings: To specify what embeddings to use, use the -e argument followed by the path to the file containing your embeddings. For example:
```
python .\lstm.py -e 'glove_embeddings/glove_25d.json'
```
We have included a folder containing formatted GloVe embeddings trained on data from Twitter.

#### Output

The code will output performance metrics and model evaluation results to the console

## BERT Models

We use a basic `bert-base-uncased` model for fine-tuning on individual datasets. The models can be fine-tuned using the `bert.ipynb` notebook and evaluated with the `bert_evaluation.ipynb` notebook.

We fine-tune the models on the OLID, Berkeley, Dynabench, and Sentiment datasets individually, as well as on the aggregated dataset including all of the four. For training and developmenet set, we use 80/20 split, and we test all models on the OLID test data.

Fine-tuned models can be found [here](https://www.dropbox.com/scl/fo/1qnowoseynjttv8c7gvk1/h?rlkey=xlqjjhdnqvzxt9j5byndopfyd&dl=0).