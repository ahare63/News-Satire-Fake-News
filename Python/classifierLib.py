"""
classifierLib.py
Author: Adam Hare <adamth@alumni.princeton.edu>
Last Updated: 4 September 2018

Description:
This file contains a number of functions used to train and evaluate data using the machine learning classifiers.
Some of these functions may be very time-intensive depending on the size of the data set. They are also configured to
run on either Princeton's Nobel or Adroit computing cluster and so some configuration changes may be required depending
on where the programs are being run.
"""

import keras.backend as k
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf


def split_data(data, frac, shuffle):
    """
    Given a `pandas` `DataFrame`, this function splits it into two parts based on the provided fraction and returns
    both.

    Args:
        data: The data to be split, in the form of a `pandas` `DataFrame`.

        frac: The fraction of data to be put in the first result. The rest will be returned in the second result. This
        number should range from 0 to 1 inclusive, where 0 puts all data in the second result and 1 puts all data in the
        first result.

        shuffle: A boolean value indicating whether or not to shuffle the data. If `False`, the data will be returned
        in the order it was received.

    Returns:
        Two `pandas` `DataFrame` objects, the first with `frac` * 100 percent of the data and the other with (1-`frac`)
        * 100 percent of the data.

    Raises:
        ValueError: Thrown if the `frac` parameter is not between 0 and 1.
   """

    # Ensure frac is a valid decimal.
    if frac < 0 or frac > 1:
        raise ValueError('Frac must be between zero and one.')
    else:
        n = len(data)
        n_first = int(n*frac)
        if shuffle:
            shuffler = np.random.permutation(n)
            return data.loc[shuffler[:n_first]], data.loc[shuffler[n_first:]]
        else:
            return data.loc[:n_first], data.loc[n_first:]


def merge_data(file_names, frac, shuffle):
    """
    This function takes a list of file paths to csv files, reads each into a `pandas` `DataFrame` object, and returns a
    `DataFrame` consisting of the data from all of these files combined. This is useful for combining data from
    different files.

    Args:
        file_names: The paths to the files to be merged as a list.

        frac: The fraction of each data in each file to be retained. This can vary from 0 to 1 inclusive, with 0
        resulting in no data being read from any of the files and 1 reading all available data.

        shuffle: A boolean value indicating whether or not the data will be shuffled. If `False`, the data will be
        returned in the order it appeared in each file, in the order the files were given in the list.

    Returns:
        One `pandas` `DataFrame` with all of the read data.

    Raises:
        IndexError: Thrown if no file_names is an empty list.

        ValueError: Thrown by `split_data` if `frac` is invalid.
   """

    try:
        file = file_names.pop(0)
        data = pd.read_csv(file, index_col=0)

        # Split the data if required. Skip this possibly expensive operation if all data is used.
        if frac != 1:
            data, _ = split_data(data, frac, shuffle)

        for f in file_names:
            new_data = pd.read_csv(f, index_col=0)

            # Split the data if required. Skip this possibly expensive operation if all data is used.
            if frac != 1:
                new_data, _ = split_data(new_data, frac, shuffle)

            # Concatenate this with the all of the data already read.
            data = pd.concat([data, new_data], ignore_index=True)
            if shuffle:
                data = data.sample(frac=1)
        return data

    except IndexError:
        print('Must include at least one data file.')


def get_bag_of_words(data, vectorizer, is_tf, use_stop_words, is_binary):
    """
    This function creates and returns a weighted bag of words from the data based on the provided parameters. `data` is
    expected to be a single column from a `pandas` `DataFrame`, typically "Body" in this use case.

    Args:

        data: A `pandas` `DataFrame` from which the bag of words will be built.

        vectorizer: The model used to make the bag of words. If `None`, a new bag of words is created by the other
        parameters. If provided, following parameters are ignored and the vectorizer is used instead.

        is_tf: A boolean value indicating whether or not to use the TF-IDF weighting. If `True`, TF-IDF is used and if
        `False` the appearances of words are simply counted.

        use_stop_words: A boolean value indicating whether or not to remove common stop words from the data. When
        `True`, common English stop words are removed and when `False` all words are considered.

        is_binary: A boolean value indicating whether or not to use a binary weighting for word appearances. If `True`,
        any words that appear at least once will get a weight of 1 and all words that don't appear a weight 0. This
        carries over to TF-IDF, but the TF-IDF still returns a range of values, only considering the word count part of
        it as binary.


    Returns:
        The bag of words data and the vectorizer used to create it.

    Raises:
        Additional errors may be thrown by dependencies.
   """

    # Check if we got a vectorizer, in which case use it and return.
    if vectorizer is not None:
        return vectorizer.transform(data), vectorizer

    # Using TF-IDF weighting
    if is_tf:
        if use_stop_words:
            vectorizer = TfidfVectorizer(stop_words='english', binary=is_binary)
        else:
            vectorizer = TfidfVectorizer(binary=is_binary)

    # Using standard word count weighting
    else:
        if use_stop_words:
            vectorizer = CountVectorizer(stop_words='english', binary=is_binary)
        else:
            vectorizer = CountVectorizer(binary=is_binary)
    return vectorizer.fit_transform(data), vectorizer


def scale_features(data, feature_list, additional_features):
    """
    This function scales and returns the appropriate feature columns.

    Args:

        data: The `pandas` `DataFrame` containing the relevant data.

        feature_list: A string list of features to be used in analysis, in this case the names of the columns in `data`
        to be used.

        additional_features: Any additional features not provided in `data` that need to be added. In this case,
        'additional_features` is often the bag of words returned by get_bag_of_words.

    Returns:
        A `numpy` `ndarray` consisting of the columns of `feature_list` scaled and concatenated with those in
        `additional_data`.

    Raises:
        Additional errors may be thrown by dependencies.
   """

    # Ensure we have some features from the data to scale.
    if feature_list:
        features = preprocessing.scale(data[list(feature_list)].values)
        return hstack([additional_features, features])
    # Otherwise, just return the additional features. Note that this list may be empty, in which case the function
    # returns an empty list.
    return additional_features


def get_precision(y_true, y_predicted):
    """
    Calculate precision. This is drawn from old `keras` source code.

    Args:
        y_true: The ground truth labels for the data set.

        y_predicted: The predicted labels for the data set, as returned by the classifier.

    Returns:
        A float signifying the classifier's precision on the given data set.

    Raises:
        Additional errors may be thrown by dependencies.
   """

    true_positives = k.sum(k.round(k.clip(y_true * y_predicted, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_predicted, 0, 1)))
    precision = true_positives / (predicted_positives + k.epsilon())
    return precision


def get_recall(y_true, y_predicted):
    """
    Calculate recall. This is drawn from old `keras` source code.

    Args:
        y_true: The ground truth labels for the data set.

        y_predicted: The predicted labels for the data set, as returned by the classifier.

    Returns:
        A float signifying the classifier's recall on the given data set.

    Raises:
        Additional errors may be thrown by dependencies.
   """

    true_positives = k.sum(k.round(k.clip(y_true * y_predicted, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + k.epsilon())
    return recall


def get_f_score(y_true, y_predicted):
    """
    Calculate F Score.

    Args:
        y_true: The ground truth labels for the data set.

        y_predicted: The predicted labels for the data set, as returned by the classifier.

    Returns:
        A float signifying the classifier's F Score on the given data set.

    Raises:
        Additional errors may be thrown by dependencies.
   """

    pre = get_precision(y_true, y_predicted)
    rec = get_recall(y_true, y_predicted)
    return (2*pre*rec)/(pre + rec)


"""
Change some settings to allow this code to work with cluster computers.
Parameters:
    None.

Returns:
    None.
"""


def config_cluster():

    # Fixes out of memory errors on Nobel.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    k.set_session(tf.Session(config=config))


"""
Calculates some important measures on the results of an SVM classification.
Parameters:
    true_labels - A vector of ground truth labels.
    
    predicted_labels - A vector of labels predicted by the classifier.
    
    iteration - Which iteration these results correspond to, for use with shuffled data. Default is `False`, which
                does not print any iteration information.
                
    verbose - A boolean indicating whether or not to print all results on separate lines. Default is `False`, which does
              not print any values in this manner.
              
    print_latex - A boolean indicating whether or not to print the data formatted for a `LaTeX` table. Default is 
                  `False`, which doesn't print anything.
Returns:
    Four floats, corresponding to the accuracy, precision, recall, and F Score of the classifier for the given data.
"""


def get_measures(true_labels, predicted_labels, iteration, verbose, print_latex):

    # Initialize variables.
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    n = len(true_labels)

    # create a count of each type of classification
    for i in range(0, n):
        if true_labels[i]:
            if predicted_labels[i]:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if predicted_labels[i]:
                false_negative += 1
            else:
                true_negative += 1

    # Check that every article was evaluated properly.
    assert (n == true_positive + false_positive + true_negative + false_negative)

    # Print the iteration number and result of each classification type
    if verbose:
        if iteration is not None:
            print("Iteration = %d" % iteration)
        print("TP = %.4f" % (true_positive / n))
        print("FP = %.4f" % (false_positive / n))
        print("FN = %.4f" % (false_negative / n))
        print("TN = %.4f" % (true_negative / n))

    # Calculate desired measures.
    accuracy = (true_positive + true_negative) / n

    # For sufficiently large and complex data sets, all of these should take some value. For testing, they don't always.
    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        print("Precision caused divide by zero error.")
        precision = 1

    # For sufficiently large and complex data sets, all of these should take some value. For testing, they don't always.
    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:
        print("Recall caused divide by zero error.")
        recall = 1

    f_score = (2 * precision * recall) / (precision + recall)

    # Print the measures if verbose.
    if verbose:
        print("Accuracy = %.4f" % accuracy)
        print("Precision = %.4f" % precision)
        print("Recall = %.4f" % recall)
        print("F = %.4f" % f_score)

    # Print in `LaTeX` format.
    if print_latex:
        # Print with iteration.
        if iteration is not None:
            print("%d & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f\\\\" %
                  (iteration, true_positive / n, false_positive / n, false_negative / n, true_negative / n, accuracy,
                   precision, recall, f_score))
        else:
            print("%.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f\\\\" %
                  (true_positive / n, false_positive / n, false_negative / n, true_negative / n, accuracy, precision,
                   recall, f_score))
        print()
    return accuracy, precision, recall, f_score
