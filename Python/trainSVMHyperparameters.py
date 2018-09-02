# Learn hyperparameters for the SVM model
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

columns = ['ARI', 'FR', 'GF', 'avgSyl', 'linkCount', 'profanityCount', 'senCount', 'titleARI',
           'titleAvgSyl', 'titleFR', 'titleGF', 'titleWordCount', 'twitChar', 'wordCount']
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

columns = ['ARI', 'FR', 'GF', 'avgSyl', 'linkCount', 'profanityCount', 'senCount', 'titleARI',
           'titleAvgSyl', 'titleFR', 'titleGF', 'titleWordCount', 'twitChar', 'wordCount']
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

