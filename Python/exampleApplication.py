"""
File: exampleApplication.py
Author: Adam Hare <adamth@alumni.princeton.edu>
Last Updated: 4 September 2018

Description:
This file contains an example application of the functions in pythonAPI.py.
"""

import pandas as pd

from pythonAPI import from_files, parse_data, preprocess_svm, train_svm_hyperparameters, build_svm, test_svm, \
    preprocess_clstm, build_clstm, test_clstm


def full_svm_test():
    """
    This function is designed to demonstrate the full functionality of the pythonAPI functions for the SVM. It reads
    data from files, builds features, formats them, builds the classifier, tests, and returns the results of the test.

    Args:
        None.

    Returns:
        None.

    Raises:
        IndexError: Can result from `from_files` function.

    """

    # Read data from files. This method is to ensure the training set has both labels.
    data_serious = from_files(["../Data/smallTrainSample.csv", "../Data/smallTrainSample.csv"])
    data_satire = from_files(["../Data/smallTrainSample.csv"])

    # Parse the desired data.
    parsed_serious = parse_data(data_serious, run_all=True, label=0, date_range=(1984, 2019))
    parsed_satire = parse_data(data_satire, run_all=True, label=1, date_range=(1984, 2019))

    # Combine the data into one set.
    data = pd.concat([parsed_serious, parsed_satire], ignore_index=True).sample(frac=1)

    # Preprocess training data.
    data, labels, vectorizer = preprocess_svm(data)

    # Learn the best hyperparameters.
    parameters = train_svm_hyperparameters(data, labels)

    # Build the classifier
    classifier = build_svm(data, labels, parameters)

    # Read testing data from files. This method is to ensure the testing set has both labels.
    test_serious = from_files(["../Data/smallTestSample.csv"])
    test_satire = from_files(["../Data/smallTestSample.csv"])

    # Parse and preprocess the test data.
    parsed_serious = parse_data(test_serious, run_all=True, label=0, date_range=(1984, 2019))
    parsed_satire = parse_data(test_satire, run_all=True, label=1, date_range=(1984, 2019))
    test_data = pd.concat([parsed_serious, parsed_satire], ignore_index=True).sample(frac=1)
    test, test_labels, _ = preprocess_svm(test_data, vectorizer)

    # Test the classifier.
    print("Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F Score: %.4f" % test_svm(classifier, test, test_labels))


def full_clstm_test():
    """
    This function is designed to demonstrate the full functionality of the pythonAPI functions for the CLSTM. It reads
    data from files, builds features, formats them, builds the classifier, tests, and returns the results of the test.

    Args:
        None.

    Returns:
        None.

    Raises:
        IndexError: Can result from `from_files` function.

    """

    # Read data from files. This method is to ensure the training set has both labels.
    data_serious = from_files(["../Data/smallTrainSample.csv", "../Data/smallTrainSample.csv"])
    data_satire = from_files(["../Data/smallTrainSample.csv"])

    # Add labels to the desired data.
    parsed_serious = parse_data(data_serious, label=0)
    parsed_satire = parse_data(data_satire, label=1)
    data = pd.concat([parsed_serious, parsed_satire], ignore_index=True).sample(frac=1)

    # Preprocess data.
    data, labels, tokenizer, max_words, max_length = preprocess_clstm(data)

    # Build the CLSTM model using `keras`.
    model = build_clstm(data, labels, max_words, max_length)

    # Read testing data from files. This method is to ensure the testing set has both labels.
    test_serious = from_files(["../Data/smallTestSample.csv"])
    test_satire = from_files(["../Data/smallTestSample.csv"])

    # Parse and preprocess the test data.
    parsed_serious = parse_data(test_serious, label=0)
    parsed_satire = parse_data(test_satire, label=1)
    test_data = pd.concat([parsed_serious, parsed_satire], ignore_index=True).sample(frac=1)
    test, test_labels, _, _ = preprocess_clstm(test_data, tokenizer, max_words=max_words, max_length=max_length)

    # Test the classifier.
    results = test_clstm(model, test, test_labels)
    print("Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F Score: %.4f" % (results[1], results[2], results[3],
                                                                            results[4]))


def main():
    """
    A main function which simply runs `full_svm_test` and `full_clstm_test`

    Args:
        None.

    Returns:
        None.

    Raises:
        IndexError: Can result from `from_files` function.

    """

    print("SVM Results:")
    full_svm_test()

    print("CLSTM Results:")
    full_clstm_test()


if __name__ == "__main__":
    main()
