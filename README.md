## Currently under construction - refactoring code
## Check out documentation [here](https://ahare63.github.io/Senior-Thesis/).
# Classifying News, Satire, and "Fake News": An SVM and Deep Learning Approach
Senior Thesis completed as part of coursework for Princeton University.
This thesis was awarded the S.S. Wilkes Memorial Prize by the Princeton Operations Research and Financial Engineering department for the best thesis on "statistics and its applications to solving societal problems."

This repository contains the final report, some of the code, and data used in my senior thesis titled *Classifying News, Satire, and "Fake News": An SVM and Deep Learning Approach*. The code is formatted to run on Princeton's Nobel and Adroit computing clusters. See the pdf for the full thesis report.

Raw data files and code for processing them is omitted. Included instead is `allData.zip`, which contains `allSatire.csv` and `allSerious.csv`. These files contain all satirical and all serious news articles used as data for this project. Also included are `train.csv` and `test.csv` which were used for training and testing satire against serious news in the project. Note that these are two divisions of the same data.

I have also included four small files that I used for testing and are helpful to see how everything works on a small amount of data. The files `smallTrainData.csv` and `smallTestData.csv` each contain a small amount of raw data, before any features have been added. `trainWithFeatures.csv` and `testWithFeatures.csv` were both generated from these files using the `parse_data` function in `python-api.py` and are ready for use in one of the classifiers.

The functions in this repository are meant to be somewhat flexible in terms of data, but require the first column to be an index of the data as well as columns named "Date", "Body", and "Title". All other columns will be added in parsing. If the data you would like to use does not have an column indicating an index, you can change the `pandas  csv_read` to generate an index for you. Many functions will work even without some of the three core columns; please read comments to determine which functions require which columns.

Almost all of this code has been refactored from what was used in the actual testing to be more readable and modular. These changes have resulted in a significant departure from the original code. This code is mostly included for reference and to give an idea of the procedures used in this thesis. This also accounts for discrepancies between code samples in the thesis and what is provided directly in this repository.

# Dependencies
This thesis makes use of several commonly used Python libraries, specifically:
* `csv`
* `keras`
* `nltk`
* `numpy`
* `pandas`
* `re`
* `scipy`
* `sklearn`
* `tensorflow`
* `text.stat` 
