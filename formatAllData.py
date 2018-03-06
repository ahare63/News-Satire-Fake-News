# Extract features that are inherent to the article and don't change based on what data set it's in
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from textstat.textstat import textstat
import re
import csv

# Best implemented with own list, stored in profaneWords.csv
with open('profaneWords.csv', 'r') as f:
    reader = csv.reader(f)
    rawList = list(reader)
    profaneList = []
    for item in rawList:
        profaneList.append(item[0])


# get the average number of syllables per word
def getAvgSyl(row, isTitle):
    if isTitle == 1:
        syls = textstat.syllable_count(row.Title)
        return syls/row.titleWordCount
    syls = textstat.syllable_count(row.Body)
    return syls/row.wordCount


# count number of profane words in text
def profanityCount(text):
    num = 0
    for word in profaneList:
        num += len(re.findall(word, text))
    return num


# count number of links/Twitter pictures
def linkCount(text):
    num = len(re.findall('http(s)?://', text))
    num += len(re.findall('pic\.twitter\.com/[A-Za-z0-9]* — .* \(@[A-Za-z0-0]*\)', text))
    return num


allSatire = pd.read_csv('./Data/combinedSets/allSatire.csv')
allSatire = allSatire.drop(['Unnamed: 0'], axis=1)
allSerious = pd.read_csv('./Data/combinedSets/allSerious.csv')
allSerious = allSerious.drop(['Unnamed: 0'], axis=1).dropna(how='any')


# Add satire label
allSatire['isSatire'] = np.repeat(1, len(allSatire.Body))
allSerious['isSatire'] = np.repeat(0, len(allSerious.Body))

# do one-hot encoding
allSatire["2010"] = allSatire["Date"].apply(lambda x: int(x == 2010))
allSerious["2010"] = allSerious["Date"].apply(lambda x: int(x == 2010))
allSatire["2011"] = allSatire["Date"].apply(lambda x: int(x == 2011))
allSerious["2011"] = allSerious["Date"].apply(lambda x: int(x == 2011))
allSatire["2012"] = allSatire["Date"].apply(lambda x: int(x == 2012))
allSerious["2012"] = allSerious["Date"].apply(lambda x: int(x == 2012))
allSatire["2013"] = allSatire["Date"].apply(lambda x: int(x == 2013))
allSerious["2013"] = allSerious["Date"].apply(lambda x: int(x == 2013))
allSatire["2014"] = allSatire["Date"].apply(lambda x: int(x == 2014))
allSerious["2014"] = allSerious["Date"].apply(lambda x: int(x == 2014))
allSatire["2015"] = allSatire["Date"].apply(lambda x: int(x == 2015))
allSerious["2015"] = allSerious["Date"].apply(lambda x: int(x == 2015))
allSatire["2016"] = allSatire["Date"].apply(lambda x: int(x == 2016))
allSerious["2016"] = allSerious["Date"].apply(lambda x: int(x == 2016))
allSatire["2017"] = allSatire["Date"].apply(lambda x: int(x == 2017))
allSerious["2017"] = allSerious["Date"].apply(lambda x: int(x == 2017))

# Get word count
allSatire['wordCount'] = allSatire['Body'].apply(lambda x: len(word_tokenize(x)))
allSatire['titleWordCount'] = allSatire['Title'].apply(lambda x: len(word_tokenize(x)))
allSerious['wordCount'] = allSerious['Body'].apply(lambda x: len(word_tokenize(x)))
allSerious['titleWordCount'] = allSerious['Title'].apply(lambda x: len(word_tokenize(x)))

# Count number of profane words
allSatire['profanityCount'] = allSatire['Body'].apply(lambda x: profanityCount(x))
allSerious['profanityCount'] = allSerious['Body'].apply(lambda x: profanityCount(x))


# Get the average number of syllables per word
allSatire['avgSyl'] = allSatire.apply(lambda x: getAvgSyl(x), axis=1)
allSatire['titleAvgSyl'] = allSatire.apply(lambda x: getAvgSyl(x, 1), axis=1)
allSerious['avgSyl'] = allSerious.apply(lambda x: getAvgSyl(x), axis=1)
allSerious['titleAvgSyl'] = allSerious.apply(lambda x: getAvgSyl(x, 1), axis=1)


# Get reading ease scores
allSatire['FR'] = allSatire['Body'].apply(lambda x: textstat.flesch_reading_ease(x))
allSatire['GF'] = allSatire['Body'].apply(lambda x: textstat.gunning_fog(x))
allSatire['ARI'] = allSatire['Body'].apply(lambda x: textstat.automated_readability_index(x))

allSatire['titleFR'] = allSatire['Title'].apply(lambda x: textstat.flesch_reading_ease(x))
allSatire['titleGF'] = allSatire['Title'].apply(lambda x: textstat.gunning_fog(x))
allSatire['titleARI'] = allSatire['Title'].apply(lambda x: textstat.automated_readability_index(x))

allSerious['FR'] = allSerious['Body'].apply(lambda x: textstat.flesch_reading_ease(x))
allSerious['GF'] = allSerious['Body'].apply(lambda x: textstat.gunning_fog(x))
allSerious['ARI'] = allSerious['Body'].apply(lambda x: textstat.automated_readability_index(x))

allSerious['titleFR'] = allSerious['Title'].apply(lambda x: textstat.flesch_reading_ease(x))
allSerious['titleGF'] = allSerious['Title'].apply(lambda x: textstat.gunning_fog(x))
allSerious['titleARI'] = allSerious['Title'].apply(lambda x: textstat.automated_readability_index(x))


# Get number of sentences
allSatire['senCount'] = allSatire['Body'].apply(lambda x: len(sent_tokenize(x)))
allSerious['senCount'] = allSerious['Body'].apply(lambda x: len(sent_tokenize(x)))

# Get number of links and Twitter characters
allSatire['linkCount'] = allSatire['Body'].apply(lambda x: linkCount(x))
allSatire['twitChar'] = allSatire['Body'].apply(lambda x: len(re.findall('[@#]', x)))

allSerious['linkCount'] = allSerious['Body'].apply(lambda x: linkCount(x))
allSerious['twitChar'] = allSerious['Body'].apply(lambda x: len(re.findall('[@#]', x)))

# Save data
allSatire.to_csv('./Data/combinedSets/allSatire.csv')
allSerious.to_csv('./Data/combinedSets/allSerious.csv')

print('Done!')
