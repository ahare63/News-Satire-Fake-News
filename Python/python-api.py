"""
python-api.py
Author: Adam Hare
Last Updated: 23 August 2018

Description:
This file contains a few higher level functions to handle data parsing, SVM testing, SVM hyper-parameter testing, and
CLSTM testing.
"""

import pandas as pd
from parserLib import get_profanity_count


"""
This function does all necessary parsing on a given csv file based on given parameters. It assumes the csv is 
formatted as in one of the provided csv files, although not all fields are needed to start. Specifically, the `Body`,
`Date`, and `Title` columns are required although they can be empty. All parsing is off by default.

    Parameters:
        filename - the name of the csv file from which to read the initial data
        profane_dict_file - the name of the csv file from which to read the dictionary of profane words
        count_profane - if True, count the number of profane words and place in 'profanityCount' column
        
        write_to_file - Whether or not to write to the input file. If `False`, no data is written. If `True`, data is 
                        written to the file specified by the filename parameter.
        target_file - If specified, a target file to write to. Otherwise, no file is written. Both write_to_file and
                      target_file can write in a given call.
                        
    Returns:
        This function returns a `pandas` `DataFrame` object with the desired parsed fields added.
"""


def parse_data(filename, profane_dict_file='profaneWords.csv', count_profane=False, write_to_file=False,
               target_file=''):

    # Read the data from the csv file.
    data = pd.read_csv(filename, index_col=0)

    # Count the number of profane words.
    if count_profane:
        data['profanityCount'] = get_profanity_count(data, profane_dict_file)

    # Write to file filename.
    if write_to_file:
        data.to_csv(filename)

    # Write to other file if specified by target_file.
    if target_file != '':
        data.to_csv(target_file)

    # Return `DataFrame` object
    return data


print(parse_data('../Data/smallTrainSample.csv'))
