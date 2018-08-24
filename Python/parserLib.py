"""
parserLib.py
Author: Adam Hare
Last Updated: 23 August 2018

Description:
This file contains a number of functions used to parse data from the csv files. Some of these functions may be very
time-intensive depending on the size of the data set.
"""

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
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



















# Function to get the average number of syllables per word.
# Here, row refers to the article being considered. isTitle is a boolean, which when True 
# calculates the average syllable count for the title. When isTitle is False, this function 
# returns the average syllable count for the body of text.
def getAvgSyl(row, isTitle):
    if isTitle == True:
        syls = textstat.syllable_count(row.Title)
        return syls/row.titleWordCount
    syls = textstat.syllable_count(row.Body)
    return syls/row.wordCount


# Count the number of links/Twitter pictures.
def linkCount(text):
    num = len(re.findall('http(s)?://', text))
    num += len(re.findall('pic\.twitter\.com/[A-Za-z0-9]* — .* \(@[A-Za-z0-0]*\)', text))
    return num

# # Read the data from the csv files.
# allSatire = pd.read_csv('../Data/allSatire.csv')
# allSatire = allSatire.drop(['Unnamed: 0'], axis=1)  # drops extra unnecessary column added by pandas
# allSerious = pd.read_csv('../Data/allSerious.csv')
# allSerious = allSerious.drop(['Unnamed: 0'], axis=1).dropna(how='any')
#
# # Add satire label. A value of one indicates that the article is satirical.
# allSatire['isSatire'] = np.repeat(1, len(allSatire.Body))
# allSerious['isSatire'] = np.repeat(0, len(allSerious.Body))
#
# # Perform one-hot encoding on the date field. A value of 1 indicates that the article was written in a given year.
# # Note that if no date is provided, these will all be 0.
# allSatire["2010"] = allSatire["Date"].apply(lambda x: int(x == 2010))
# allSerious["2010"] = allSerious["Date"].apply(lambda x: int(x == 2010))
# allSatire["2011"] = allSatire["Date"].apply(lambda x: int(x == 2011))
# allSerious["2011"] = allSerious["Date"].apply(lambda x: int(x == 2011))
# allSatire["2012"] = allSatire["Date"].apply(lambda x: int(x == 2012))
# allSerious["2012"] = allSerious["Date"].apply(lambda x: int(x == 2012))
# allSatire["2013"] = allSatire["Date"].apply(lambda x: int(x == 2013))
# allSerious["2013"] = allSerious["Date"].apply(lambda x: int(x == 2013))
# allSatire["2014"] = allSatire["Date"].apply(lambda x: int(x == 2014))
# allSerious["2014"] = allSerious["Date"].apply(lambda x: int(x == 2014))
# allSatire["2015"] = allSatire["Date"].apply(lambda x: int(x == 2015))
# allSerious["2015"] = allSerious["Date"].apply(lambda x: int(x == 2015))
# allSatire["2016"] = allSatire["Date"].apply(lambda x: int(x == 2016))
# allSerious["2016"] = allSerious["Date"].apply(lambda x: int(x == 2016))
# allSatire["2017"] = allSatire["Date"].apply(lambda x: int(x == 2017))
# allSerious["2017"] = allSerious["Date"].apply(lambda x: int(x == 2017))
#
# # Get word count on both the body and title.
# allSatire['wordCount'] = allSatire['Body'].apply(lambda x: len(word_tokenize(x)))
# allSatire['titleWordCount'] = allSatire['Title'].apply(lambda x: len(word_tokenize(x)))
# allSerious['wordCount'] = allSerious['Body'].apply(lambda x: len(word_tokenize(x)))
# allSerious['titleWordCount'] = allSerious['Title'].apply(lambda x: len(word_tokenize(x)))
#
# # Count the number of profane words.
# allSatire['profanityCount'] = allSatire['Body'].apply(lambda x: profanityCount(x))
# allSerious['profanityCount'] = allSerious['Body'].apply(lambda x: profanityCount(x))
#
# # Get the average number of syllables per word for both the body and title.
# allSatire['avgSyl'] = allSatire.apply(lambda x: getAvgSyl(x, False), axis=1)
# allSatire['titleAvgSyl'] = allSatire.apply(lambda x: getAvgSyl(x, True), axis=1)
# allSerious['avgSyl'] = allSerious.apply(lambda x: getAvgSyl(x, False), axis=1)
# allSerious['titleAvgSyl'] = allSerious.apply(lambda x: getAvgSyl(x, True), axis=1)
#
# # Get reading ease scores for both the body and title.
# allSatire['FR'] = allSatire['Body'].apply(lambda x: textstat.flesch_reading_ease(x))
# allSatire['GF'] = allSatire['Body'].apply(lambda x: textstat.gunning_fog(x))
# allSatire['ARI'] = allSatire['Body'].apply(lambda x: textstat.automated_readability_index(x))
#
# allSatire['titleFR'] = allSatire['Title'].apply(lambda x: textstat.flesch_reading_ease(x))
# allSatire['titleGF'] = allSatire['Title'].apply(lambda x: textstat.gunning_fog(x))
# allSatire['titleARI'] = allSatire['Title'].apply(lambda x: textstat.automated_readability_index(x))
#
# allSerious['FR'] = allSerious['Body'].apply(lambda x: textstat.flesch_reading_ease(x))
# allSerious['GF'] = allSerious['Body'].apply(lambda x: textstat.gunning_fog(x))
# allSerious['ARI'] = allSerious['Body'].apply(lambda x: textstat.automated_readability_index(x))
#
# allSerious['titleFR'] = allSerious['Title'].apply(lambda x: textstat.flesch_reading_ease(x))
# allSerious['titleGF'] = allSerious['Title'].apply(lambda x: textstat.gunning_fog(x))
# allSerious['titleARI'] = allSerious['Title'].apply(lambda x: textstat.automated_readability_index(x))
#
# # Get number of sentences.
# allSatire['senCount'] = allSatire['Body'].apply(lambda x: len(sent_tokenize(x)))
# allSerious['senCount'] = allSerious['Body'].apply(lambda x: len(sent_tokenize(x)))
#
# # Get number of links and Twitter characters.
# allSatire['linkCount'] = allSatire['Body'].apply(lambda x: linkCount(x))
# allSatire['twitChar'] = allSatire['Body'].apply(lambda x: len(re.findall('[@#]', x)))
#
# allSerious['linkCount'] = allSerious['Body'].apply(lambda x: linkCount(x))
# allSerious['twitChar'] = allSerious['Body'].apply(lambda x: len(re.findall('[@#]', x)))
#
# # Write the data back to the original csv files.
# allSatire.to_csv('./Data/combinedSets/allSatire.csv')
# allSerious.to_csv('./Data/combinedSets/allSerious.csv')
#
# print('Done!')
