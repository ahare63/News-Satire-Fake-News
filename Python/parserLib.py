"""
parserLib.py
Author: Adam Hare <adamth@alumni.princeton.edu>
Last Updated: 6 September 2018

Description:
This file contains a number of functions used to parse data from the csv files. Some of these functions may be very
time-intensive depending on the size of the data set.
"""

from textstat.textstat import textstat
import re
import csv


def profanity_count(text, profane_list):
    """
    This function counts the number of profane words in a given text.

    Args:

        text: The text to be considered as a string.

        profane_list: The list of words considered to be profane.

    Returns:
        An integer indicating the number of profane words found in the text.

    Raises:
        Additional errors may be thrown by dependencies.
   """

    num = 0
    for word in profane_list:
        num += len(re.findall(word, text))
    return num


def get_profanity_count(data, dict_file):
    """
    Opens the csv that contains the profane word dictionary and count the number of appearances for each article.

    Args:

        data: The data to run the profanity count on.

        dict_file: A path to a csv that contains a list of profane words.

    Returns:
        A column that has a count of the number of profane words found in the "Body" column.

    Raises:
        Additional errors may be thrown by dependencies.
   """

    with open(dict_file, 'r') as file:
        profane_list = []
        for item in list(csv.reader(file)):
            profane_list.append(item[0])
        return data['Body'].apply(lambda x: profanity_count(x, profane_list))


def get_encoded_date(data, date_range):
    """
    Does one-hot encoding on the date. Assigns a one to the column corresponding to the year provided and a one to all
    others. If the date is `NaN` or out of range, it leaves all columns as zero. Optionally takes a range of supported
    years, by default 2010-2017 inclusive.

    Args:

        data: The data to encode the date on. In this case a `pandas` `DataFrame` with a column called "Date".

        date_range: A range of dates to encode. Includes the lower bound but not the upper bound.

    Returns:
        `data` with new columns for the one-hot encoded data.

    Raises:
        Additional errors may be thrown by dependencies.
   """

    for date in date_range:
        data[str(date)] = data["Date"].apply(lambda x: int(x == date))
    return data


def get_avg_syl_count(row, is_title):
    """
    Function to get the average number of syllables per word. Here, row refers to the article being considered.

    Args:

        row: The row of data to be considered. In this case a row of a `pandas` `DataFrame`.

        is_title: A boolean value indicating whether or not this average syllable count is for the title.

    Returns:
        An average syllable count for the provided row.

    Raises:
        Additional errors may be thrown by dependencies.
   """

    if is_title:
        syl = textstat.syllable_count(row.Title)
        return syl/row.titleWordCount
    syl = textstat.syllable_count(row.Body)
    return syl/row.wordCount


def get_link_count(text):
    """
    Function to count the number of links and Twitter pictures in the provided text.

    Args:

        text: The text to be analyzed.

    Returns:
        An integer indicating the number of links found.

    Raises:
        Additional errors may be thrown by dependencies.
   """

    num = len(re.findall('http(s)?://', text))
    num += len(re.findall('pic\.twitter\.com/[A-Za-z0-9]* — .* \(@[A-Za-z0-9]*\)', text))
    return num
