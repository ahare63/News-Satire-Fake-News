"""
parserLib.py
Author: Adam Hare
Last Updated: 23 August 2018

Description:
This file contains a number of functions used to parse data from the csv files. Some of these functions may be very
time-intensive depending on the size of the data set.
"""

from textstat.textstat import textstat
import re
import csv


# Count number of profane words in a given text.
def profanity_count(text, profane_list):
    num = 0
    for word in profane_list:
        num += len(re.findall(word, text))
    return num


# Open the csv that contains the profane word dictionary and count the number of appearances for each article.
def get_profanity_count(data, dict_file):
    with open(dict_file, 'r') as file:
        profane_list = []
        for item in list(csv.reader(file)):
            profane_list.append(item[0])
        return data['Body'].apply(lambda x: profanity_count(x, profane_list))


# Do one-hot encoding on the date. Assigns a one to the column corresponding to the year provided and a one to all
# others. If the date is NaN or out of range, it leaves all columns as zero. Optionally takes a range of supported
# years, by default 2010-2017 inclusive.
def get_encoded_date(data, date_range):
    for date in date_range:
        data[str(date)] = data["Date"].apply(lambda x: int(x == date))
    return data


# Function to get the average number of syllables per word.
# Here, row refers to the article being considered. is_title is a boolean, which when `True`
# calculates the average syllable count for the title. When isTitle is False, this function
# returns the average syllable count for the body of text.
def get_avg_syl_count(row, is_title):
    if is_title:
        syls = textstat.syllable_count(row.Title)
        return syls/row.titleWordCount
    syls = textstat.syllable_count(row.Body)
    return syls/row.wordCount


# Count the number of links and Twitter pictures in the provided text.
def get_link_count(text):
    num = len(re.findall('http(s)?://', text))
    num += len(re.findall('pic\.twitter\.com/[A-Za-z0-9]* — .* \(@[A-Za-z0-0]*\)', text))
    return num
