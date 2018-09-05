"""
pythonAPI.py
Author: Adam Hare <adamth@alumni.princeton.edu>
Last Updated: 4 September 2018

Description:
This file contains a few higher level functions to handle data parsing, SVM testing, SVM hyper-parameter testing, and
CLSTM testing.
"""

from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Conv1D, LSTM, Dense
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import re
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from textstat.textstat import textstat

from classifierLib import config_cluster, get_bag_of_words, get_f_score, get_precision, get_recall, merge_data, \
    scale_features, get_measures
from parserLib import get_avg_syl_count, get_encoded_date, get_link_count, get_profanity_count


"""
This function takes a list of files as input. It reads those files into `pandas` `DataFrame` objects and combines them.

Parameters:
    files - A list of paths to the files to be read.

    percentage - The percentage of each file to be read. By default 1, which reads all available data. Should range
                 between 0 and 1. Set this to a lower number for debugging classifiers.

    shuffle - A boolean value indicating whether or not to shuffle the data. Default is `True`, which shuffles all data.
"""


def from_files(files, frac=1, shuffle=True):
    return merge_data(files, frac, shuffle)


"""
This function does all necessary parsing on a given `pandas` `DataFrame` based on given parameters. It assumes the 
`DataFrame` is formatted as in one of the provided csv files, although not all fields are needed to start. Specifically,
the `Body`, `Date`, and `Title` columns are required although they can be empty. All parsing is off by default.

    Parameters:
        # Note: all boolean parameters are `False` by default. This indicates that no calculation will be made. If this
        # function is passed only the filename parameter, it will do nothing. Additionally, running each of these will 
        # overwrite whatever is in the column they write to but not change any other columns.
    
        data - A `pandas` `DataFrame` with the relevant data.
                        
        target_file - If specified, a target file to write to. Otherwise, no file is written. Default is `None`,
                      indicating no data will be written.
                      
        run_all - A boolean value indicating whether or not to build all features. Overrides all subsequent parameters
                  except profane_dict_file and date_range. By default `False`, meaning that all other parameters are
                  followed as specified.
                      
        count_profane - If `True`, count the number of profane words and place in 'profanityCount' column.
        
        profane_dict_file - The name of the csv file from which to read the dictionary of profane words. By default,
                            'profaneWords.csv'. Setting this has no effect if count_profane=False. This csv should 
                            contain only regular expressions of what will be identified as profane words. See included 
                            file `profaneWords.csv` for an example.
                            
        label - If specified, indicates whether or not an article is satire. Default is `None`, which skips adding a
                    label. If 0, every article in the file is given a label of 0 for non-satirical. If 1, every article
                    is given a label of 1 for satirical. Throws an error if given something other than 0 or 1.
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


def parse_data(data, target_file=None, run_all=False, count_profane=False, profane_dict_file='profaneWords.csv',
               label=None, encode_date=False, date_range=range(2010, 2018), title_word_count=False,
               word_count=False, title_syl_count=False, body_syl_count=False, sentence_count=False, link_count=False,
               twitter_count=False, title_fr_score=False, fr_score=False, title_gf_score=False, gf_score=False,
               title_ari_score=False, ari_score=False):

    # Check to see if the run_all flag has been set to True.
    if run_all:
        print("Warning: all features to be run. This may overwrite values in existing columns.")
        return parse_data(data, target_file=target_file, count_profane=True, profane_dict_file=profane_dict_file,
                          label=label, encode_date=True, date_range=date_range, title_word_count=True,
                          word_count=True, title_syl_count=True, body_syl_count=True, sentence_count=True,
                          link_count=True, twitter_count=True, title_fr_score=True, fr_score=True, title_gf_score=True,
                          gf_score=True, title_ari_score=True, ari_score=True)

    # Count the number of profane words.
    if count_profane:
        data['profanityCount'] = get_profanity_count(data, profane_dict_file)

    # Add satirical label.
    if label is not None:
        data['isSatire'] = np.repeat(label, len(data.Body))

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

    # Write to file if specified by target_file.
    if target_file is not None:
        data.to_csv(target_file)

    # Return `DataFrame` object
    return data


"""
This function preprocesses the data, preparing it for use by an SVM classifier. It returns a `pandas` `DataFrame` 
containing the processed data, a `numpy` `ndarray` of labels for that data, and the vectorizer used to build the bag of
words.
    Parameters:

        data - A `pandas` `DataFrame` object containing the data to be preprocessed.
        
        vectorizer - If it exists, the vectorizer used on the training data. This is to ensure that the training and 
                     testing data were built using the same vectorizer. When supplied, the function assumes that this is
                     testing data and builds using the already established vectorizer. By default this is `None`, which
                     causes the function to assume this is testing data and build a new vectorizer. Including this will
                     cause the parameters `is_tf`, `use_stop_words`, and `is_binary` to be ignored if supplied because
                     they will already have been set when the vectorizer was initialized.

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

        feature_columns - A list of strings indicating column names to be used as features. They will all be normalized.
                          By default, this is all values used in testing. Pass an empty list to include no additional
                          features.

        params - Hyperparameters to be tested. By default, these are `class_weight` as "balanced" and `None` and `C`
                 from 10^-5 to 10^3. Please refer to the `sk-learn` documentation for more information.


    Returns:
        This function returns a `pandas` `DataFrame` containing the processed data, a `numpy` `ndarray` of labels 
        for that data, and the bag of words vectorizer.
"""


def preprocess_svm(data, vectorizer=None, label_column="isSatire", bag_of_words_column="Body", is_tf=False,
                   use_stop_words=True, is_binary=True, feature_columns=None):

    # Get data labels.
    labels = data[label_column].values

    # Build the bag of words.
    bag_of_words, vectorizer = get_bag_of_words(data[bag_of_words_column], vectorizer, is_tf, use_stop_words, is_binary)

    # Extract the relevant features.
    if feature_columns is None:
        feature_columns = ['ARI', 'FR', 'GF', 'avgSyl', 'linkCount', 'profanityCount', 'senCount', 'titleARI',
                           'titleAvgSyl', 'titleFR', 'titleGF', 'titleWordCount', 'twitChar', 'wordCount']
    return scale_features(data, feature_columns, bag_of_words), labels, vectorizer


"""
This function preprocesses the data, preparing it for use by the CLSTM classifier. It returns a `pandas` `DataFrame` 
containing the processed data, a `numpy` `ndarray` of labels for that data, the tokenizer, and the max_length parameter.
    Parameters:

        data - A `pandas` `DataFrame` object containing the data to be preprocessed.
        
        tokenizer - A tokenizer to use to preprocess the data. If not included, this function will assume the data is
                    for training and build a new tokenizer. If provided, the function will assume this is testing data
                    and build based on the provided tokenizer. It is important that the parameters `append_title`, 
                    `max_words`, and `max_length` are the same for training and testing. By default this is `None` and a 
                    new tokenizer is built.

        label_column - A string indicating the name of the of the column to be used as the labels for the data. By 
                       default, this is "isSatire".
                       
        append_title - A boolean value indicating whether or not to append the article title to the front of the body
                       so that it is included in the analysis. By default, this is `True`, so the title is appended.

        max_words - An int indicating the maximum size of the dictionary used in tokenization. The top max_words used
                    in the corpus (based on frequency) will be retained. By default, this is `None`, which uses all
                    available words. For a sufficiently large corpus, this can dramatically and unnecessarily increase
                    compute time. Recommended max size for the corpus used in this thesis about 20,000.
                    
        max_length - The maximum length of an article to consider. Articles longer than this will be truncated and
                     shorter articles will be padded with null. By default this is the length of the longest article.
                     Again, for a significantly large corpus this increases compute time dramatically without a
                     performance benefit. Look at the distribution of article length in your corpus to set. Recommended
                     at 4000 for the corpus used in this thesis.


    Returns:
    It returns a `pandas` `DataFrame` containing the processed data, a `numpy` `ndarray` of labels for that data, the 
    tokenizer, the max_words, and the max_length parameter. `max_words` and `max_length` are returned in case they are
    computed relative to the data because they are needed when preprocessing the testing data.
"""


def preprocess_clstm(data, tokenizer=None, label_column="isSatire", append_title=True, max_words=None, max_length=None):

    # Get data labels.
    labels = data[label_column].values

    # Append the title to the body if specified.
    if append_title:
        data['Body'] = data.apply(lambda x: x.Title + ". " + x.Body, axis=1)

    # Check if we already have a tokenizer to use.
    if tokenizer is not None:
        as_sequence = tokenizer.texts_to_sequences(data.Body)
        test = np.array(pad_sequences(as_sequence, maxlen=max_length))  # Pad for use with CLSTM.
        test[test > max_words] = 0  # Replace any words not in the dictionary with nulls.
        return test, labels, tokenizer, max_length

    # Otherwise, build a new tokenizer.
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data.Body)
    as_sequence = tokenizer.texts_to_sequences(data.Body)

    # If not provided, calculate the maximum length.
    if max_length is None:
        max_length = len(max(as_sequence, key=len))

    # If not provided, calculate the maximum number of words.
    if max_words is None:
        max_words = len(tokenizer.word_index)

    # Pad for use with CLSTM and return.
    return np.array(pad_sequences(as_sequence, maxlen=max_length)), labels, tokenizer, max_words, max_length


"""
This function learns hyper-parameters for the SVM for a specific data set and a specific range of parameters.
This function makes heavy use of standard `sk-learn` functions.
    Parameters:
        data - A `pandas` `DataFrame` object containing the preprocessed data.

        labels - The labels for data.

        params - Hyperparameters to be tested. By default, these are `class_weight` as "balanced" and `None` and `C`
                 from 10^-5 to 10^3. Please refer to the `sk-learn` documentation for more information.

        scoring - The metrics to be used for judging classifier efficiency, as specified in `sk-learn` documentation.
                  The default is 'accuracy'.

        verbose - A boolean value indicating whether or not to print all of the data returned from the results. Default
                  is `False`, which doesn't print anything.
    Returns:
        A dictionary of the hyperparameters found to fit best with this data.
"""


def train_svm_hyperparameters(data, labels, params=None, scoring='accuracy', verbose=False):

    # Set the parameters to be tried.
    if params is None:
        params = {'class_weight': ['balanced', None],
                  'C': [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 10 ** 2, 10 ** 3]}

    # Use the sk-learn library to find the best hyperparameters.
    svc = LinearSVC()
    classifier = GridSearchCV(svc, params, scoring=scoring)
    classifier.fit(data, labels)

    # Print the results.
    if verbose:
        print(classifier.cv_results_)
        print("Best parameters: ", classifier.best_params_)
        print("Best score: ", classifier.best_score_)
        print("Best estimator: ", classifier.best_estimator_)

    return classifier.best_params_


"""
This function uses the provided data to build an SVM classifier.
    Parameters:

        data - A `pandas` `DataFrame`, likely returned from either `preprocess_svm` or preprocess_nn` on which the
               classification will be performed.
        
        labels - A `numpy` `ndarray` containing the labels for the data.
                         
                          
        params - Hyperparameters to be used in classification.
                 
    Returns:
        This function returns a classifier trained on the provided data.
"""


def build_svm(data, labels, params):

    # Build using the specified parameters.
    svc = LinearSVC(**params)

    # Fit on the provided data and return.
    return svc.fit(data, labels)


"""
This function uses the provided data to build a CLSTM classifier. The default values provided reflect the best 
performance  on the data set used for this thesis.
    Parameters:

        data - A `pandas` `DataFrame`, likely returned from either `preprocess_svm` or preprocess_nn` on which the
               classification will be performed.

        labels - A `numpy` `ndarray` containing the labels for the data.


        max_words - The max_words of the embedding, as specified in preprocessing.
        
        max_length - The maximum length of the embedding, as specified in preprocessing.
        
        is_cluster - A boolean value indicating whether or not to configure this for a cluster computer. Default is
                     `False` and the configuration is not changed.
                     
        embedding_size - An integer indicating the size of the embedding layer. Default is 100.
        
        dropout - A decimal indicating the dropout value. Default is 0.3.
        
        filters - An integer indicating the number of filters in the convolutional layer. Default is 64.
        
        kernel - An integer indicating the kernel size of the convolutional layer. Default is 3.
        
        conv_activation - A string indicating the type of activation function for the convolutional layer. Default is
                          'relu'.
                          
        lstm_units - An integer indicating the number of units to use in the LSTM. Default is 64.
        
        dense_activation - A string indicating the type of activation function to use for the dense layer. Default is
                           'sigmoid'.
                           
        metrics - A list indicating the metrics to use when evaluating classifier performance. Default is to use
                  accuracy, precision, recall, and F Score.
                  
        early_stopping - A boolean indicating whether or not to stop early if the classifier performs worse on a given
                         epoch. Default is `True`, so the training will stop if performance degrades.
                         
        batch_size - An integer indicating the batch size to use in training. Default is 512.
        
        epochs - An integer indicating the number of epochs to use in training. Default is 10. Fewer epochs may be used
                 in practice if early_stopping is `True`.
                 
        shuffle - A boolean indicating whether or not to shuffle the data. Default is `True`.
        
        class_weight - A dictionary of weights for different classes. This can boost performance in cases where one 
                       class is much bigger than the other.

    Returns:
        This function returns a classifier trained on the provided data.
"""


def build_clstm(data, labels, max_words, max_length, is_cluster=False, embedding_size=100, dropout=0.3, filters=64,
                kernel=3, conv_activation='relu', lstm_units=64, dense_activation='sigmoid', metrics=None,
                early_stopping=True, batch_size=512, epochs=10, shuffle=True, class_weight=None):

    # Set configuration for cluster computer if indicated.
    if is_cluster:
        config_cluster()

    # Build the CLSTM model with Keras.
    model_clstm = Sequential()
    model_clstm.add(Embedding(max_words + 1, embedding_size, input_length=max_length))
    model_clstm.add(Dropout(dropout))
    model_clstm.add(Conv1D(filters=filters, kernel_size=kernel, activation=conv_activation))
    model_clstm.add(LSTM(units=lstm_units))
    model_clstm.add(Dense(1, activation=dense_activation))

    # Fill in default metrics if none provided.
    if metrics is None:
        metrics = ['accuracy', get_precision, get_recall, get_f_score]

    # Compile the model.
    model_clstm.compile(loss='binary_crossentropy', optimizer='Adam', metrics=metrics)

    # Set early stopping if instructed.
    if early_stopping:
        callbacks = [EarlyStopping(monitor='loss')]
    else:
        callbacks = None

    # Fit and return the CLSTM.
    model_clstm.fit(data, y=np.array(labels), callbacks=callbacks, batch_size=batch_size, epochs=epochs,
                    shuffle=shuffle, class_weight=class_weight)
    return model_clstm


"""
This function finally evaluates the CLSTM on the test data.
Parameters: 
    model - The CLSTM model to be evaluated on.
    
    data - The data to be tested on.
    
    labels - The true labels for the testing data.
    
    verbose - A boolean indicating whether or not to print the results. By default `False` and no results are printed.
    
    print_latex - A boolean indicating whether or not to print the results formatted for a LaTeX table. Default is 
                  `False` which does not print results in this format.
                  
Returns:
    Returns the results.
"""


def test_clstm(model, data, labels, verbose=False, print_latex=False):

    # Evaluate the data and print the results.
    results = model.evaluate(data, np.array(labels))
    if verbose:
        print(results)

    # Print formatted for LaTeX table.
    if print_latex:
        print(" %.4f & %.4f & %.4f & %.4f \\\\" % (results[1], results[2], results[3], results[4]))

    return results


"""
This function finally evaluates the SVM model on the test data.
Parameters:
    model - The SVM model to be evaluated on.
    
    data - The data to be tested on.
    
    labels - The true labels for the testing data.
    
    iteration - An integer corresponding to the iteration number this is. By default, `None`.
    
    verbose - A boolean indicating whether or not all data should be printed. By default, `False` and no data is 
              printed.
              
    print_latex - A boolean indicating whether or not to print all data as a row in a `LaTeX` table. Default is `False`
                  and no data is printed.
                  
Returns:
    Returns accuracy, precision, recall, and F Score for the tested data.
    
"""


def test_svm(model, data, labels, verbose=False, print_latex=False):

    # Predict labels.
    predicted_labels = model.predict(data)

    # Get performance measures.
    return get_measures(labels, predicted_labels, iteration=None, verbose=verbose, print_latex=print_latex)
