"""
classifierLib.py
Author: Adam Hare
Last Updated: 24 August 2018

Description:
This file contains a number of functions used to train and evaluate data using the machine learning classifiers.
Some of these functions may be very time-intensive depending on the size of the data set. They are also configured to
run on either Princeton's Nobel or Adroit computing cluster and so some configuration changes may be required depending
on where the programs are being run.
"""

import numpy as np
import pandas as pd


# Given a `pandas` `DataFrame`, this function splits it into two parts based on the provided percentage and returns
# both as a tuple. Valid values for percentage range from 0 to 1 inclusive. If shuffle is `True`, all data is shuffled,
# otherwise it is returned in the order it was read.
def split_data(data, percentage, shuffle):
    if percentage < 0 or percentage > 1:
        raise ValueError('Percentage must be between zero and one.')
    else:
        n = len(data)
        n_first = int(n*percentage)
        if shuffle:
            shuffler = np.random.permutation(n)
            return data.loc[shuffler[:n_first]], data.loc[shuffler[n_first:]]
        else:
            return data.loc[:n_first], data.loc[n_first:]


# This function takes a list of file paths to csv files, reads each into a `pandas` `DataFrame` object, and returns a
# `DataFrame` consisting of the data from all of these files combined. This is useful for combining data from different
# files. Before combining files, it reduces them to a percentage specified by the percentage parameter. This ensures
# that each file is proportionally represented if not all data is needed by the function calling merge_data. If shuffle
# is set to `False`, all data will be maintained in order.
def merge_data(file_names, percentage, shuffle):

    try:
        file = file_names.pop(0)
        data = pd.read_csv(file, index_col=0)

        # Split the data if required. Skip this possibly expensive operation if all data is used.
        if percentage != 1:
            data, _ = split_data(data, percentage, shuffle)

        for f in file_names:
            new_data = pd.read_csv(f, index_col=0)

            # Split the data if required. Skip this possibly expensive operation if all data is used.
            if percentage != 1:
                new_data, _ = split_data(new_data, percentage, shuffle)

            # Concatenate this with the all of the data already read.
            data = pd.concat([data, new_data], ignore_index=True)
            if shuffle:
                data = data.sample(frac=1)

        return data

    except IndexError:
        print('Must include at least one data file.')

