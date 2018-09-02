"""
python-api.py
Author: Adam Hare <adamth@alumni.princeton.edu>
Last Updated: 2 September 2018

Description:
This file contains a few higher level functions to handle data parsing, SVM testing, SVM hyper-parameter testing, and
CLSTM testing.
"""

from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import pandas as pd
import re
from textstat.textstat import textstat

from classifierLib import get_bag_of_words, merge_data
from parserLib import get_avg_syl_count, get_encoded_date, get_link_count, get_profanity_count


"""
This function does all necessary parsing on a given csv file based on given parameters. It assumes the csv is 
formatted as in one of the provided csv files, although not all fields are needed to start. Specifically, the `Body`,
`Date`, and `Title` columns are required although they can be empty. All parsing is off by default.

    Parameters:
        # Note: all boolean parameters are `False` by default. This indicates that no calculation will be made. If this
        # function is passed only the filename parameter, it will build and return a `pandas` `DataFrame` read from the
        # csv specified by filename. Additionally, running each of these will overwrite whatever is in the column they
        # write to but not change any other columns.
    
        filename - The name of the csv file from which to read the initial data.
        
        write_to_file - Whether or not to write to the input file. If `False`, no data is written. If `True`, data is 
                        written to the file specified by the filename parameter. `False` by default.
                        
        target_file - If specified, a target file to write to. Otherwise, no file is written. Both write_to_file and
                      target_file can write in a given call.
                      
        run_all - A boolean value indicating whether or not to build all features. Overrides all subsequent parameters
                  except profane_dict_file and date_range. By default `False`, meaning that all other parameters are
                  followed as specified.
                      
        count_profane - If `True`, count the number of profane words and place in 'profanityCount' column.
        
        profane_dict_file - The name of the csv file from which to read the dictionary of profane words. By default,
                            'profaneWords.csv'. Setting this has no effect if count_profane=False. This csv should 
                            contain only regular expressions of what will be identified as profane words. See included 
                            file `profaneWords.csv` for an example.
                            
        is_satire - If specified, indicates whether or not an article is satire. Default is -1, which skips adding a
                    label. If 0, every article in the file is given a label of 0 for non-satirical. If 1, every article
                    is given a label of 1 for satirical. Throws an error if given something other than -1, 0, or 1.
                    Writes to 'isSatire' column.
                    
        encode_date - A boolean value indicating whether or not to do one-hot encoding on the date field. `True` does
                      the encoding. Writes to columns with the year as the name.
                      
        date_range - A range of dates to consider. Only valid if encode_date=True. Default is 2010-2017 inclusive.
        
        title_word_count - A boolean indicating whether or not to generate a word count for the title. Writes to 
                           'titleWordCount' column if `True`.
                           
        word_count - A boolean indicating whether or not to generate a word count for the body. Writes to 'wordCount' 
                     column if `True`.
                     
        title_syl_count - A boolean indicating whether or not to generate a title average syllable count. Writes to 
                         'titleAvgSyl' column if `True`. Only valid if titleWordCount column is populated.
                         
        body_syl_count - A boolean indicating whether or not to generate a body average syllable count.  Writes to 
                         'avgSyl' column if `True`. Only valid if wordCount column is populated.
                         
        sentence_count - A boolean indicating whether or not to generate a count of the number of sentences in the body.
                         Writes to 'senCount' column if `True`.
                         
        link_count - A boolean indicating whether or not to generate a count of the number of links found in the body
                     of the articles.  Writes to 'linkCount' column if `True`.
                     
        twitter_count - A boolean indicating whether or not to generate a count of the number of Twitter characters
                        found in the body. Writes to 'twitChar' column if `True`.
                        
        title_fr_score - A boolean indicating whether or not to run Flesch Reading Ease scoring algorithm on the title. 
                         Writes to 'titleFR' column if `True`.
                         
        fr_score - A boolean indicating whether or not to run Flesch Reading Ease scoring algorithm on the body. 
                   Writes to 'FR' column if `True`.
                        
        title_gf_score - A boolean indicating whether or not to run the Gunning Fog scoring algorithm on the title. 
                         Writes to 'titleGF' column if `True`.
                         
        gf_score - A boolean indicating whether or not to run the Gunning Fog scoring algorithm on the body. 
                   Writes to 'GF' column if `True`.
                   
        title_ari_score - A boolean indicating whether or not to run the Automated Readability Index algorithm on the 
                          title. Writes to 'titleARI' column if `True`.
                         
        ari_score - A boolean indicating whether or not to run the Automated Readability Index algorithm on the body. 
                    Writes to 'ARI' column if `True`.
                        
    Returns:
        This function returns a `pandas` `DataFrame` object with the desired parsed fields added.
"""


def parse_data(filename, write_to_file=False, target_file='', run_all=False, count_profane=False,
               profane_dict_file='profaneWords.csv', is_satire=-1, encode_date=False, date_range=range(2010, 2018),
               title_word_count=False, word_count=False, title_syl_count=False, body_syl_count=False,
               sentence_count=False, link_count=False, twitter_count=False, title_fr_score=False, fr_score=False,
               title_gf_score=False, gf_score=False, title_ari_score=False, ari_score=False):

    # Check to see if the run_all flag has been set to True.
    if run_all:
        print("Warning: all features to be run. This may overwrite values in existing columns.")
        return parse_data(filename, write_to_file=write_to_file, target_file=target_file, count_profane=True,
                          profane_dict_file=profane_dict_file, is_satire=is_satire, encode_date=True,
                          date_range=date_range, title_word_count=True, word_count=True, title_syl_count=True,
                          body_syl_count=True, sentence_count=True, link_count=True, twitter_count=True,
                          title_fr_score=True, fr_score=True, title_gf_score=True, gf_score=True, title_ari_score=True,
                          ari_score=True)

    # Read the data from the csv file.
    data = pd.read_csv(filename, index_col=0)

    # Count the number of profane words.
    if count_profane:
        data['profanityCount'] = get_profanity_count(data, profane_dict_file)

    # Add satirical label.
    if is_satire == 0 or is_satire == 1:
        data['isSatire'] = np.repeat(is_satire, len(data.Body))
    elif is_satire != -1:
        raise ValueError('Invalid value for is_satire: should be -1, 0, or 1')

    # Do one-hot encoding on the date.
    if encode_date:
        get_encoded_date(data, date_range)
    elif date_range != range(2010, 2018):
        raise ValueError('Date range cannot be set if encode_date=False')

    # Get word counts.
    if title_word_count:
        data['titleWordCount'] = data['Title'].apply(lambda x: len(word_tokenize(x)))
    if word_count:
        data['wordCount'] = data['Body'].apply(lambda x: len(word_tokenize(x)))

    # Do average syllable counts.
    if title_syl_count:
        if 'titleWordCount' not in data.columns:
            raise IndexError('Must generate title word count before calculating title average syllable count ')
        data['titleAvgSyl'] = data.apply(lambda x: get_avg_syl_count(x, False), axis=1)
    if body_syl_count:
        if 'wordCount' not in data.columns:
            raise IndexError('Must generate word count before calculating average syllable count ')
        data['avgSyl'] = data.apply(lambda x: get_avg_syl_count(x, True), axis=1)

    # Get sentence counts.
    if sentence_count:
        data['senCount'] = data['Body'].apply(lambda x: len(sent_tokenize(x)))

    # Get the number of links in the body.
    if link_count:
        data['linkCount'] = data['Body'].apply(lambda x: get_link_count(x))

    # Get the number of Twitter characters in the body
    if twitter_count:
        data['twitChar'] = data['Body'].apply(lambda x: len(re.findall('[@#]', x)))

    # Generate Flesch Reading Ease score.
    if title_fr_score:
        data['titleFR'] = data['Title'].apply(lambda x: textstat.flesch_reading_ease(x))
    if fr_score:
        data['FR'] = data['Body'].apply(lambda x: textstat.flesch_reading_ease(x))

    # Generate Gunning Fog score.
    if title_gf_score:
        data['titleGF'] = data['Title'].apply(lambda x: textstat.gunning_fog(x))
    if gf_score:
        data['GF'] = data['Body'].apply(lambda x: textstat.gunning_fog(x))

    # Generate Automated Readability Index score.
    if title_ari_score:
        data['titleARI'] = data['Title'].apply(lambda x: textstat.automated_readability_index(x))
    if ari_score:
        data['ARI'] = data['Body'].apply(lambda x: textstat.automated_readability_index(x))

    # Write to file filename.
    if write_to_file:
        data.to_csv(filename)

    # Write to other file if specified by target_file.
    if target_file != '':
        data.to_csv(target_file)

    # Return `DataFrame` object
    return data


"""
This function learns hyper-parameters for the SVM for a specific data set and a specific range of parameters.
This function makes heavy use of standard sklearn functions.
    Parameters:

        training_files - A list of names of the csv files from which to read the initial training data. These files
                         will be combined to form the training set.
                         
        training_percentage - The percentage of the available training data to be trained on. One may wish to reduce 
                              this to decrease compute time. Valid from 0 to 1, inclusive, although a value of zero will 
                              not produce meaningful results. Generally, this should be as close to 1 as resources will
                              allow. Default is 1.
                              
        shuffle - A boolean value indicating whether or not to shuffle the data. Default is `True`, meaning the data is
                  shuffled.
                  
        label_column - A string indicating the name of the of the column to be used as the labels for the data. By 
                       default, this is "isSatire".
                       
        bag_of_words_column - A string indicating the name of the column to be used as the source for the bag of words
                              data. By default, this is "Body".
                              
        is_tf - A boolean value indicating whether or not to use the TF-IDF weighting. Default is `False`, which means
                that a simple word count will be used instead.
        
        use_stop_words - A boolean value indicating whether or not to use English "stop words." Default is `True`, 
                         which means that common English stop words will be removed from analysis.
        
        is_binary - A boolean value indicating whether or not to use a binary weighting. Default is `True`, which means
                    that all words will be given a value of 0 if they do not appear in a given text and 1 if they appear
                    at least once.


    Returns:
        This function returns a `pandas` `DataFrame` object with the desired parsed fields added.
"""


def train_hyperparameters(training_files, training_percentage=1, shuffle=True, label_column="isSatire",
                          bag_of_words_column="Body", is_tf=False, use_stop_words=True, is_binary=True):
    # Read data from specified files
    data = merge_data(training_files, training_percentage, shuffle)

    # Get data labels
    labels = data[label_column].values

    # Build the bag of words
    bag_of_words = get_bag_of_words(data[bag_of_words_column], is_tf, use_stop_words, is_binary)

    return "Done"


print(parse_data("../Data/smallTestSample.csv", write_to_file=False, target_file="../Data/testWithFeatures.csv",
                 run_all=True, is_satire=0, date_range=range(1984, 2019)))
