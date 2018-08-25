# Learn hyperparameters for the SVM model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack
from sklearn import preprocessing

vectorizer = CountVectorizer(stop_words='english', binary=True)
binaryBog = vectorizer.fit_transform(train.Body)
print('Vectorized train')

labels = train["isSatire"].values
columns = ['ARI', 'FR', 'GF', 'avgSyl', 'linkCount', 'profanityCount', 'senCount', 'titleARI',
           'titleAvgSyl', 'titleFR', 'titleGF', 'titleWordCount', 'twitChar', 'wordCount']
features = preprocessing.scale(train[list(columns)].values)
features = hstack([binaryBog, features])
param = {'class_weight': ['balanced', None], 'C': [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3]}

svc = LinearSVC()
clf = GridSearchCV(svc, param, scoring='accuracy')
clf.fit(features, labels)
print(clf.cv_results_)
print(clf.best_estimator_)
print(clf.best_params_)
print(clf.best_score_)
print('All features done')

features = binaryBog
param = {'class_weight': ['balanced', None], 'C': [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3]}
svc = LinearSVC()
clf = GridSearchCV(svc, param, scoring='accuracy')
clf.fit(features, labels)
print(clf.cv_results_)
print(clf.best_estimator_)
print(clf.best_params_)
print(clf.best_score_)
print('Binary BOW Done')

vectorizer = TfidfVectorizer(stop_words='english', binary=True)
tfBog = vectorizer.fit_transform(train.Body)
print('Vectorized train')

labels = train["isSatire"].values
columns = ['ARI', 'FR', 'GF', 'avgSyl', 'linkCount', 'profanityCount', 'senCount', 'titleARI',
           'titleAvgSyl', 'titleFR', 'titleGF', 'titleWordCount', 'twitChar', 'wordCount']
features = preprocessing.scale(train[list(columns)].values)
features = hstack([tfBog, features])
param = {'class_weight': ['balanced', None], 'C': [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3]}

svc = LinearSVC()
clf = GridSearchCV(svc, param, scoring='accuracy')
clf.fit(features, labels)
print(clf.cv_results_)
print(clf.best_estimator_)
print(clf.best_params_)
print(clf.best_score_)
print('All features done')

features = tfBog
param = {'class_weight': ['balanced', None], 'C': [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3]}
svc = LinearSVC()
clf = GridSearchCV(svc, param, scoring='accuracy')
clf.fit(features, labels)
print(clf.cv_results_)
print(clf.best_estimator_)
print(clf.best_params_)
print(clf.best_score_)
print('TF-IDF BOW Done')

