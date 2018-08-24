"""
SVMTestNobel.py
Author: Adam Hare
Last Updated: 23 August 2018

Description:
This script runs the SVM model on Princeton's Nobel cluster.
"""


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
import numpy as np
from sklearn import preprocessing

# print some important measures (accuracy, precision, recall, F measure)
# here, x is a list of the predicted values returned from the SVM, y is a list of the "ground truth" values
# scoreList is a structure for storing results to be used later, iteration indicates which iteration of the algorithm is being analyzed
# verbose is a boolean which prints nicely formatted data if true or just the iteration and LaTeX formatted data if false
def printMeasures(x, y, scoreList, iteration, verbose=False):
    # initialize variables
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0
    N = len(x)
    
    # create a count of each type of classification
    for i in range(0, N):
        if x[i]:
            if y[i]:
                TP += 1
            else:
                FP += 1
        else:
            if y[i]:
                FN += 1
            else:
                TN += 1

    assert(N == TP+FP+TN+FN)  # Check that every article was evaluated

    # store each result as a percentage
    scoreList[0][iteration] = TP/N
    scoreList[1][iteration] = FP/N
    scoreList[2][iteration] = FN/N
    scoreList[3][iteration] = TN/N

    # print the iteration number, optionally print the percentage of each classification type
    if verbose
        print("Iteration = %d" % iteration)
        print("TP = %.4f" % (TP/N))
        print("FP = %.4f" % (FP/N))
        print("FN = %.4f" % (FN/N))
        print("TN = %.4f" % (TN/N))

    # calculate desired measures
    acc = (TP+TN)/N
    pre = TP/(TP + FP)
    rec = TP/(TP + FN)
    fm = (2*pre*rec)/(pre + rec)

    # store measures
    scoreList[4][iteration] = acc
    scoreList[5][iteration] = pre
    scoreList[6][iteration] = rec
    scoreList[7][iteration] = fm

    if verbose
        print("Accuracy = %.4f" % acc)
        print("Precision = %.4f" % pre)
        print("Recall = %.4f" % rec)
        print("F = %.4f" % fm)
        
    print("%d & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f\\\\" % (iteration, TP/N, FP/N, FN/N, TN/N, acc,
                                                                              pre, rec, fm))
    print()


# train and test on entire corpus, serious vs satire
def testAllBin(numIts, featureList):
    # lists to hold: true positive, false positive, true negative, false negative, accuracy, precision, recall, F measure
    scoreList = [[None] * numIts for _ in range(8)]

    allSatire = pd.read_csv('./Data/allSatire.csv')
    allSatire = allSatire.drop(['Unnamed: 0'], axis=1)

    allSerious = pd.read_csv('./Data/allSerious.csv')
    allSerious = allSerious.drop(['Unnamed: 0'], axis=1)

    for i in range(0, numIts):
        N = len(allSatire.Body)
        Ntrain = int(0.85 * N)
        shuffler = np.random.permutation(N)
        satireTrain = allSatire.loc[shuffler[:Ntrain]]
        satireTest = allSatire.loc[shuffler[Ntrain:]]

        N = len(allSerious.Body)
        Ntrain = int(0.85 * N)
        shuffler = np.random.permutation(N)
        seriousTrain = allSerious.loc[shuffler[:Ntrain]]
        seriousTest = allSerious.loc[shuffler[Ntrain:]]

        train = pd.concat([seriousTrain, satireTrain], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)
        test = pd.concat([seriousTest, satireTest], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)

        if featureList[0]:
            vectorizer = CountVectorizer(stop_words='english', binary=True)
            bogTrain = vectorizer.fit_transform(train.Body)
            print('Vectorized train')

            bogTest = vectorizer.transform(test.Body)
            print('Vectorized test')

            svc = LinearSVC(C=0.01, class_weight=None)

        else:
            vectorizer = TfidfVectorizer(stop_words='english', binary=True)
            bogTrain = vectorizer.fit_transform(train.Body)
            print('Vectorized train')

            bogTest = vectorizer.transform(test.Body)
            print('Vectorized test')

            svc = LinearSVC(C=1, class_weight='balanced')

        labels = train["isSatire"].values
        columns = []
        if featureList[1]:
            columns += ['titleAvgSyl', 'titleFR', 'titleGF', 'titleWordCount', 'titleARI']
        if featureList[2]:
            columns += ['profanityCount']
        if featureList[3]:
            columns += ['twitChar']
        if featureList[4]:
            columns += ['linkCount']
        if featureList[5]:
            columns += ['ARI', 'FR', 'GF']
        if featureList[6]:
            columns += ['avgSyl', 'senCount', 'wordCount']
        if featureList[7]:
            columns += ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

        if columns:
            features = preprocessing.scale(train[list(columns)].values)
            features = hstack([bogTrain, features])
        else:  # if we want only the bag of words
            features = bogTrain

        svc.fit(features, labels)
        print('Trained')

        if columns:
            Xtest = preprocessing.scale(test[list(columns)].values)
            Xtest = hstack([bogTest, Xtest])
        else:
            Xtest = bogTest

        Ytest = test["isSatire"].values
        yT = svc.predict(Xtest)

        printMeasures(yT, Ytest, scoreList, i)

    TP = scoreList[0]
    print("Avg TP = %.4f" % np.mean(TP))
    FP = scoreList[1]
    print("Avg FP = %.4f" % np.mean(FP))
    FN = scoreList[2]
    print("Avg FN = %.4f" % np.mean(FN))
    TN = scoreList[3]
    print("Avg TN = %.4f" % np.mean(TN))
    Acc = scoreList[4]
    print("Avg Acc = %.4f" % np.mean(Acc))
    Pre = scoreList[5]
    print("Avg Pre = %.4f" % np.mean(Pre))
    Rec = scoreList[6]
    print("Avg Rec = %.4f" % np.mean(Rec))
    F = scoreList[7]
    print("Avg F = %.4f" % np.mean(F))
    for iter in range(0, numIts):
        print('\hline')
        print("%d & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f\\\\" % (iter, TP[iter], FP[iter], FN[iter],
                                                                                  TN[iter], Acc[iter], Pre[iter],
                                                                                  Rec[iter], F[iter]))


# train and test on entire corpus, serious vs "fake news"
def testSeriousFN(numIts, featureList):
    # lists to hold: true positive, false positive, true negative, false negative, accuracy, precision, recall, F measure
    scoreList = [[None] * numIts for _ in range(8)]

    allSatire = pd.read_csv('./Data/allSatire.csv')
    allSatire = allSatire.drop(['Unnamed: 0'], axis=1)
    NR = allSatire[allSatire.Source == 'NR']
    WN = allSatire[allSatire.Source == 'WN']
    HU = allSatire[allSatire.Source == 'HU']
    EN = allSatire[allSatire.Source == 'EN']
    fakeNews = pd.concat([NR, WN, HU, EN], ignore_index=True)
    print(len(fakeNews.Body))

    for i in range(0, numIts):
        allSerious = pd.read_csv('./Data/allSerious.csv')
        allSerious = allSerious.drop(['Unnamed: 0'], axis=1)
        allSerious = allSerious.sample(n=30370).reset_index(drop=True)

        N = len(fakeNews.Body)
        Ntrain = int(0.85 * N)
        shuffler = np.random.permutation(N)
        fnTrain = fakeNews.loc[shuffler[:Ntrain]]
        fnTest = fakeNews.loc[shuffler[Ntrain:]]

        N = len(allSerious.Body)
        Ntrain = int(0.85 * N)
        shuffler = np.random.permutation(N)
        seriousTrain = allSerious.loc[shuffler[:Ntrain]]
        seriousTest = allSerious.loc[shuffler[Ntrain:]]

        train = pd.concat([seriousTrain, fnTrain], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)
        test = pd.concat([seriousTest, fnTest], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)

        if featureList[0]:
            vectorizer = CountVectorizer(stop_words='english', binary=True)
            bogTrain = vectorizer.fit_transform(train.Body)
            print('Vectorized train')

            bogTest = vectorizer.transform(test.Body)
            print('Vectorized test')

            svc = LinearSVC(C=0.01, class_weight=None)

        else:
            vectorizer = TfidfVectorizer(stop_words='english', binary=True)
            bogTrain = vectorizer.fit_transform(train.Body)
            print('Vectorized train')

            bogTest = vectorizer.transform(test.Body)
            print('Vectorized test')

            svc = LinearSVC(C=1, class_weight='balanced')

        labels = train["isSatire"].values
        columns = []
        if featureList[1]:
            columns += ['titleAvgSyl', 'titleFR', 'titleGF', 'titleWordCount', 'titleARI']
        if featureList[2]:
            columns += ['profanityCount']
        if featureList[3]:
            columns += ['twitChar']
        if featureList[4]:
            columns += ['linkCount']
        if featureList[5]:
            columns += ['ARI', 'FR', 'GF']
        if featureList[6]:
            columns += ['avgSyl', 'senCount', 'wordCount']
        if featureList[7]:
            columns += ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

        if columns:
            features = preprocessing.scale(train[list(columns)].values)
            features = hstack([bogTrain, features])
        else:  # if we want only the bag of words
            features = bogTrain

        svc.fit(features, labels)
        print('Trained')

        if columns:
            Xtest = preprocessing.scale(test[list(columns)].values)
            Xtest = hstack([bogTest, Xtest])
        else:
            Xtest = bogTest

        Ytest = test["isSatire"].values
        yT = svc.predict(Xtest)

        printMeasures(yT, Ytest, scoreList, i)

    TP = scoreList[0]
    print("Avg TP = %.4f" % np.mean(TP))
    FP = scoreList[1]
    print("Avg FP = %.4f" % np.mean(FP))
    FN = scoreList[2]
    print("Avg FN = %.4f" % np.mean(FN))
    TN = scoreList[3]
    print("Avg TN = %.4f" % np.mean(TN))
    Acc = scoreList[4]
    print("Avg Acc = %.4f" % np.mean(Acc))
    Pre = scoreList[5]
    print("Avg Pre = %.4f" % np.mean(Pre))
    Rec = scoreList[6]
    print("Avg Rec = %.4f" % np.mean(Rec))
    F = scoreList[7]
    print("Avg F = %.4f" % np.mean(F))
    for iter in range(0, numIts):
        print('\hline')
        print("%d & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f\\\\" % (iter, TP[iter], FP[iter], FN[iter],
                                                                                  TN[iter], Acc[iter], Pre[iter],
                                                                                  Rec[iter], F[iter]))


# train and test on entire corpus, satire vs "fake news"
def testSatFN(numIts, featureList):
    # lists to hold: true positive, false positive, true negative, false negative, accuracy, precision, recall, F measure
    scoreList = [[None] * numIts for _ in range(8)]

    allSatire = pd.read_csv('./Data/allSatire.csv')
    allSatire = allSatire.drop(['Unnamed: 0'], axis=1)
    NR = allSatire[allSatire.Source == 'NR']
    WN = allSatire[allSatire.Source == 'WN']
    HU = allSatire[allSatire.Source == 'HU']
    EN = allSatire[allSatire.Source == 'EN']
    fakeNews = pd.concat([NR, WN, HU, EN], ignore_index=True)
    fakeNews["isFake"] = 1

    justSatire = allSatire[allSatire.Source != 'NR']
    justSatire = justSatire[justSatire.Source != 'WN']
    justSatire = justSatire[justSatire.Source != 'HU']
    justSatire = justSatire[justSatire.Source != 'EN']
    justSatire = justSatire.sample(frac=1).reset_index(drop=True)
    justSatire["isFake"] = 0

    for i in range(0, numIts):

        N = len(fakeNews.Body)
        Ntrain = int(0.85 * N)
        shuffler = np.random.permutation(N)
        fnTrain = fakeNews.loc[shuffler[:Ntrain]]
        fnTest = fakeNews.loc[shuffler[Ntrain:]]

        N = len(justSatire.Body)
        Ntrain = int(0.85 * N)
        shuffler = np.random.permutation(N)
        satireTrain = justSatire.loc[shuffler[:Ntrain]]
        satireTest = justSatire.loc[shuffler[Ntrain:]]

        train = pd.concat([satireTrain, fnTrain], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)
        test = pd.concat([satireTest, fnTest], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)

        if featureList[0]:
            vectorizer = CountVectorizer(stop_words='english', binary=True)
            bogTrain = vectorizer.fit_transform(train.Body)
            print('Vectorized train')

            bogTest = vectorizer.transform(test.Body)
            print('Vectorized test')

            svc = LinearSVC(C=0.01, class_weight=None)

        else:
            vectorizer = TfidfVectorizer(stop_words='english', binary=True)
            bogTrain = vectorizer.fit_transform(train.Body)
            print('Vectorized train')

            bogTest = vectorizer.transform(test.Body)
            print('Vectorized test')

            svc = LinearSVC(C=1, class_weight='balanced')

        labels = train["isFake"].values
        columns = []
        if featureList[1]:
            columns += ['titleAvgSyl', 'titleFR', 'titleGF', 'titleWordCount', 'titleARI']
        if featureList[2]:
            columns += ['profanityCount']
        if featureList[3]:
            columns += ['twitChar']
        if featureList[4]:
            columns += ['linkCount']
        if featureList[5]:
            columns += ['ARI', 'FR', 'GF']
        if featureList[6]:
            columns += ['avgSyl', 'senCount', 'wordCount']
        if featureList[7]:
            columns += ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

        if columns:
            features = preprocessing.scale(train[list(columns)].values)
            features = hstack([bogTrain, features])
        else:  # if we want only the bag of words
            features = bogTrain

        svc.fit(features, labels)
        print('Trained')

        if columns:
            Xtest = preprocessing.scale(test[list(columns)].values)
            Xtest = hstack([bogTest, Xtest])
        else:
            Xtest = bogTest

        Ytest = test["isFake"].values
        yT = svc.predict(Xtest)

        printMeasures(yT, Ytest, scoreList, i)

    TP = scoreList[0]
    print("Avg TP = %.4f" % np.mean(TP))
    FP = scoreList[1]
    print("Avg FP = %.4f" % np.mean(FP))
    FN = scoreList[2]
    print("Avg FN = %.4f" % np.mean(FN))
    TN = scoreList[3]
    print("Avg TN = %.4f" % np.mean(TN))
    Acc = scoreList[4]
    print("Avg Acc = %.4f" % np.mean(Acc))
    Pre = scoreList[5]
    print("Avg Pre = %.4f" % np.mean(Pre))
    Rec = scoreList[6]
    print("Avg Rec = %.4f" % np.mean(Rec))
    F = scoreList[7]
    print("Avg F = %.4f" % np.mean(F))
    for iter in range(0, numIts):
        print('\hline')
        print("%d & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f\\\\" % (iter, TP[iter], FP[iter], FN[iter],
                                                                                  TN[iter], Acc[iter], Pre[iter],
                                                                                  Rec[iter], F[iter]))


# train and test on 2010-2013 and 2014-2017
def testSP1(numIts, featureList, trainSet, testSet):  # trainSet and testSet should be either 2010 or 2014
    # lists to hold: true positive, false positive, true negative, false negative, accuracy, precision, recall, F measure
    scoreList = [[None] * numIts for _ in range(8)]

    allSatire = pd.read_csv('./Data/allSatire.csv')
    allSatire = allSatire.drop(['Unnamed: 0'], axis=1).dropna(how='any', subset={'Date'})

    allSerious = pd.read_csv('./Data/allSerious.csv')
    allSerious = allSerious.drop(['Unnamed: 0'], axis=1).dropna(how='any', subset={'Date'})

    sameSet = 0

    if trainSet == 2010 and testSet == 2010:
        allSatire = allSatire[allSatire.Date <= 2013].reset_index(drop=True)
        allSerious = allSerious[allSerious.Date <= 2013].reset_index(drop=True)
        sameSet = 1

    elif trainSet == 2014 and testSet == 2014:
        allSatire = allSatire[allSatire.Date > 2013].reset_index(drop=True)
        allSerious = allSerious[allSerious.Date > 2013].reset_index(drop=True)
        sameSet = 1

    for i in range(0, numIts):

        if sameSet:
            N = len(allSatire.Body)
            Ntrain = int(0.85 * N)
            shuffler = np.random.permutation(N)
            satTrain = allSatire.loc[shuffler[:Ntrain]]
            satTest = allSatire.loc[shuffler[Ntrain:]]

            N = len(allSerious.Body)
            Ntrain = int(0.85 * N)
            shuffler = np.random.permutation(N)
            serTrain = allSerious.loc[shuffler[:Ntrain]]
            serTest = allSerious.loc[shuffler[Ntrain:]]

            train = pd.concat([satTrain, serTrain], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)
            test = pd.concat([satTest, serTest], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)

        elif trainSet == 2010:
            trainSat = allSatire[allSatire.Date <= 2013]
            trainSer = allSerious[allSerious.Date <= 2013]
            train = pd.concat([trainSat, trainSer], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)

            testSat = allSatire[allSatire.Date > 2013]
            testSer = allSerious[allSerious.Date > 2013]
            test = pd.concat([testSat, testSer], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)

        else:
            trainSat = allSatire[allSatire.Date > 2013]
            trainSer = allSerious[allSerious.Date > 2013]
            train = pd.concat([trainSat, trainSer], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)

            testSat = allSatire[allSatire.Date <= 2013]
            testSer = allSerious[allSerious.Date <= 2013]
            test = pd.concat([testSat, testSer], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)

        if featureList[0]:
            vectorizer = CountVectorizer(stop_words='english', binary=True)
            bogTrain = vectorizer.fit_transform(train.Body)
            print('Vectorized train')

            bogTest = vectorizer.transform(test.Body)
            print('Vectorized test')

            svc = LinearSVC(C=0.01, class_weight=None)

        else:
            vectorizer = TfidfVectorizer(stop_words='english', binary=True)
            bogTrain = vectorizer.fit_transform(train.Body)
            print('Vectorized train')

            bogTest = vectorizer.transform(test.Body)
            print('Vectorized test')

            svc = LinearSVC(C=1, class_weight='balanced')

        labels = train["isSatire"].values
        columns = []
        if featureList[1]:
            columns += ['titleAvgSyl', 'titleFR', 'titleGF', 'titleWordCount', 'titleARI']
        if featureList[2]:
            columns += ['profanityCount']
        if featureList[3]:
            columns += ['twitChar']
        if featureList[4]:
            columns += ['linkCount']
        if featureList[5]:
            columns += ['ARI', 'FR', 'GF']
        if featureList[6]:
            columns += ['avgSyl', 'senCount', 'wordCount']
        if featureList[7]:
            columns += ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

        if columns:
            features = preprocessing.scale(train[list(columns)].values)
            features = hstack([bogTrain, features])
        else:  # if we want only the bag of words
            features = bogTrain

        svc.fit(features, labels)
        print('Trained')

        if columns:
            Xtest = preprocessing.scale(test[list(columns)].values)
            Xtest = hstack([bogTest, Xtest])
        else:
            Xtest = bogTest

        Ytest = test["isSatire"].values
        yT = svc.predict(Xtest)

        printMeasures(yT, Ytest, scoreList, i)

    TP = scoreList[0]
    print("Avg TP = %.4f" % np.mean(TP))
    FP = scoreList[1]
    print("Avg FP = %.4f" % np.mean(FP))
    FN = scoreList[2]
    print("Avg FN = %.4f" % np.mean(FN))
    TN = scoreList[3]
    print("Avg TN = %.4f" % np.mean(TN))
    Acc = scoreList[4]
    print("Avg Acc = %.4f" % np.mean(Acc))
    Pre = scoreList[5]
    print("Avg Pre = %.4f" % np.mean(Pre))
    Rec = scoreList[6]
    print("Avg Rec = %.4f" % np.mean(Rec))
    F = scoreList[7]
    print("Avg F = %.4f" % np.mean(F))
    for iter in range(0, numIts):
        print('\hline')
        print("%d & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f\\\\" % (iter, TP[iter], FP[iter], FN[iter],
                                                                                  TN[iter], Acc[iter], Pre[iter],
                                                                                  Rec[iter], F[iter]))


# train and test on 2014-2015 and 2016-2017
def testSP2(numIts, featureList, trainSet, testSet):  # trainSet and testSet should be either 2014 or 2016
    # lists to hold: true positive, false positive, true negative, false negative, accuracy, precision, recall, F measure
    scoreList = [[None] * numIts for _ in range(8)]

    allSatire = pd.read_csv('./Data/allSatire.csv')
    allSatire = allSatire.drop(['Unnamed: 0'], axis=1).dropna(how='any', subset={'Date'})

    allSerious = pd.read_csv('./Data/allSerious.csv')
    allSerious = allSerious.drop(['Unnamed: 0'], axis=1).dropna(how='any', subset={'Date'})

    sameSet = 0

    if trainSet == 2014 and testSet == 2014:
        allSatire = allSatire[allSatire.Date <= 2015].reset_index(drop=True)
        allSerious = allSerious[allSerious.Date <= 2015].reset_index(drop=True)
        sameSet = 1

    elif trainSet == 2016 and testSet == 2016:
        allSatire = allSatire[allSatire.Date > 2015].reset_index(drop=True)
        allSerious = allSerious[allSerious.Date > 2015].reset_index(drop=True)
        sameSet = 1

    for i in range(0, numIts):

        if sameSet:
            N = len(allSatire.Body)
            Ntrain = int(0.85 * N)
            shuffler = np.random.permutation(N)
            satTrain = allSatire.loc[shuffler[:Ntrain]]
            satTest = allSatire.loc[shuffler[Ntrain:]]

            N = len(allSerious.Body)
            Ntrain = int(0.85 * N)
            shuffler = np.random.permutation(N)
            serTrain = allSerious.loc[shuffler[:Ntrain]]
            serTest = allSerious.loc[shuffler[Ntrain:]]

            train = pd.concat([satTrain, serTrain], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)
            test = pd.concat([satTest, serTest], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)

        elif trainSet == 2014:
            trainSat = allSatire[allSatire.Date <= 2015]
            trainSer = allSerious[allSerious.Date <= 2015]
            train = pd.concat([trainSat, trainSer], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)

            testSat = allSatire[allSatire.Date > 2015]
            testSer = allSerious[allSerious.Date > 2015]
            test = pd.concat([testSat, testSer], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)

        else:
            trainSat = allSatire[allSatire.Date > 2015]
            trainSer = allSerious[allSerious.Date > 2015]
            train = pd.concat([trainSat, trainSer], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)

            testSat = allSatire[allSatire.Date <= 2015]
            testSer = allSerious[allSerious.Date <= 2015]
            test = pd.concat([testSat, testSer], ignore_index=True).dropna(how='any', subset={'Body'}).sample(frac=1)

        if featureList[0]:
            vectorizer = CountVectorizer(stop_words='english', binary=True)
            bogTrain = vectorizer.fit_transform(train.Body)
            print('Vectorized train')

            bogTest = vectorizer.transform(test.Body)
            print('Vectorized test')

            svc = LinearSVC(C=0.01, class_weight=None)

        else:
            vectorizer = TfidfVectorizer(stop_words='english', binary=True)
            bogTrain = vectorizer.fit_transform(train.Body)
            print('Vectorized train')

            bogTest = vectorizer.transform(test.Body)
            print('Vectorized test')

            svc = LinearSVC(C=1, class_weight='balanced')

        labels = train["isSatire"].values
        columns = []
        if featureList[1]:
            columns += ['titleAvgSyl', 'titleFR', 'titleGF', 'titleWordCount', 'titleARI']
        if featureList[2]:
            columns += ['profanityCount']
        if featureList[3]:
            columns += ['twitChar']
        if featureList[4]:
            columns += ['linkCount']
        if featureList[5]:
            columns += ['ARI', 'FR', 'GF']
        if featureList[6]:
            columns += ['avgSyl', 'senCount', 'wordCount']
        if featureList[7]:
            columns += ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

        if columns:
            features = preprocessing.scale(train[list(columns)].values)
            features = hstack([bogTrain, features])
        else:  # if we want only the bag of words
            features = bogTrain

        svc.fit(features, labels)
        print('Trained')

        if columns:
            Xtest = preprocessing.scale(test[list(columns)].values)
            Xtest = hstack([bogTest, Xtest])
        else:
            Xtest = bogTest

        Ytest = test["isSatire"].values
        yT = svc.predict(Xtest)

        printMeasures(yT, Ytest, scoreList, i)

    TP = scoreList[0]
    print("Avg TP = %.4f" % np.mean(TP))
    FP = scoreList[1]
    print("Avg FP = %.4f" % np.mean(FP))
    FN = scoreList[2]
    print("Avg FN = %.4f" % np.mean(FN))
    TN = scoreList[3]
    print("Avg TN = %.4f" % np.mean(TN))
    Acc = scoreList[4]
    print("Avg Acc = %.4f" % np.mean(Acc))
    Pre = scoreList[5]
    print("Avg Pre = %.4f" % np.mean(Pre))
    Rec = scoreList[6]
    print("Avg Rec = %.4f" % np.mean(Rec))
    F = scoreList[7]
    print("Avg F = %.4f" % np.mean(F))
    for iter in range(0, numIts):
        print('\hline')
        print("%d & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f\\\\" % (iter, TP[iter], FP[iter], FN[iter],
                                                                                  TN[iter], Acc[iter], Pre[iter],
                                                                                  Rec[iter], F[iter]))


# Which features to include: zeroth = BIN, first = T, second = PR, third = TW, fourth = L, fifth = RI, sixth = TC
# seventh is one-hot encoding date (DO)
featureList = [0, 1, 1, 1, 1, 1, 1, 0]
numIts = 10  # Number of iterations
testSP2(numIts, featureList, 2016, 2014)  # train on 2016-2017 and test on 2014-2015
