import re

import csv
import numpy as np
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "#"
                           "]+", flags=re.UNICODE)
user_pattern = re.compile("@USER")
ampersand_pattern = re.compile("&amp")
non_alpha_pattern = re.compile("[^a-zA-Z -]")

stemmer = PorterStemmer()
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


def preprocess_tweet(tweet, word_root_method):
    tweet = emoji_pattern.sub(r' ', tweet)
    tweet = user_pattern.sub(r'', tweet)
    tweet = tweet.lower()
    tweet = tweet.strip()
    tweet = ampersand_pattern.sub(r'', tweet)
    tweet = non_alpha_pattern.sub(r'', tweet)
    if word_root_method == 'stemming':
        tweet = stemmer.stem(tweet)
    elif word_root_method == 'lemmatization':
        tweet = lemmatizer.lemmatize(tweet, pos='v')
    return tweet


def preprocess(data_file_path, subtask='subtask_a'):
    data = []  
    with open(data_file_path, encoding="utf8") as f:
        reader = csv.DictReader(f, dialect='excel-tab')
        for row in reader:
            if row[subtask] == 'NULL':
                continue
            row['tweet'] = preprocess_tweet(row['tweet'], 'lemmatization')
            data.append(row)
    return data


def preprocess_test_set(data_file_path):
    data = []
    with open(data_file_path, encoding="utf8") as f:
        reader = csv.DictReader(f, dialect='excel-tab')
        for row in reader:
            row['tweet'] = preprocess_tweet(row['tweet'], 'lemmatization')
            data.append(row)
    return data
             

def get_corpus(data):
  return [row['tweet'] for row in data]


def get_labels(data, subtask):
    y = []

    for row in data:
        if subtask == 'subtask_a':
            y.append(1 if row['subtask_a'] == 'OFF' else 0)
        elif subtask == 'subtask_b':
            y.append(1 if row['subtask_b'] == 'TIN' else 0)
        elif subtask == 'subtask_c':
            y.append(2 if row['subtask_c'] == 'IND' else
                     (1 if row['subtask_c'] == 'GRP' else 0))
    return y


def get_feature_vectors(corpus, feature_extractor, mode='tf-idf'):
  X = []
  for row in corpus:
    transformed = np.array(feature_extractor.tfidf_transform(row).todense())
    transformed = np.reshape(transformed, transformed.shape[1])
    X.append(transformed)
  return X


def cross_validation_split(training_data):
    holdout = int(3/4 * len(training_data))
    actual_training_data = training_data[ : holdout]
    validation_data = training_data[holdout+1:]
    return actual_training_data, validation_data


def split_and_parse(training_data, subtask):
    """
    Splits data into 3/4 training set and 1/4 holdout set, and extracts the corpus and labels.
    :param training_data: a list of OrderedDict, representing our initial training data
    :param subtask: the subtask for which to select labels
    :return: 4 lists: training corpus, holdout corpus, training labels, holdout labels
    """
    actual_training_data, validation_data = cross_validation_split(training_data)
    training_corpus = get_corpus(actual_training_data)
    holdout_corpus = get_corpus(validation_data)
    training_labels = get_labels(actual_training_data, subtask)
    holdout_labels = get_labels(validation_data, subtask)
    return training_corpus, holdout_corpus, training_labels, holdout_labels


def parse(data, subtask):
    """
    Gets corpus and labels from data.
    """
    corpus = get_corpus(data)
    labels = get_labels(data, subtask)
    return corpus, labels
