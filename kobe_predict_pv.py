import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, AdaBoostClassifier

sns.set_style('white')
# Load Data
data = pd.read_csv('kobedata.csv')
# Setup Data
data['remaining_time'] = data['minutes_remaining'] * 60 + data['seconds_remaining']
data['home_play'] = pd.Series(1, index=data.index)
_ = data.set_value(data.matchup.str.contains('@')==True, 'home_play', 0)

train = data[(~data.shot_made_flag.isnull())]
test = data[(data.shot_made_flag.isnull())]

#
#print train.isnull().sum()
#print "-----"
#print test.isnull().sum()

data_scrubbed = data.copy()
target = data_scrubbed['shot_made_flag'].copy()

# Remove some columns
data_scrubbed.drop('team_id', axis=1, inplace=True) # Always one number
data_scrubbed.drop('lat', axis=1, inplace=True) # Correlated with loc_x
data_scrubbed.drop('lon', axis=1, inplace=True) # Correlated with loc_y
data_scrubbed.drop('game_id', axis=1, inplace=True) # Independent
data_scrubbed.drop('team_name', axis=1, inplace=True) # Always LA Lakers
data_scrubbed.drop('shot_made_flag', axis=1, inplace=True)
# Time
data_scrubbed['last_5_sec_in_period'] = data_scrubbed['remaining_time'] < 5
data_scrubbed.drop('minutes_remaining', axis=1, inplace=True)
data_scrubbed.drop('seconds_remaining', axis=1, inplace=True)
data_scrubbed.drop('remaining_time', axis=1, inplace=True)
# Game date
data_scrubbed['game_date'] = pd.to_datetime(data_scrubbed['game_date'])
data_scrubbed['game_year'] = data_scrubbed['game_date'].dt.year
data_scrubbed['game_month'] = data_scrubbed['game_date'].dt.month
data_scrubbed['game_day'] = data_scrubbed['game_date'].dt.day
data_scrubbed.drop('game_date', axis=1, inplace=True)
# Loc_x, and loc_y binning
data_scrubbed['loc_x'] = pd.cut(data_scrubbed['loc_x'], 25)
data_scrubbed['loc_y'] = pd.cut(data_scrubbed['loc_y'], 25)
#
data_scrubbed.drop('matchup', axis=1, inplace=True)
# Replace 20 least common action types with value 'Other'
rare_action_types = data_scrubbed['action_type'].value_counts().sort_values().index.values[:20]
data_scrubbed.loc[data_scrubbed['action_type'].isin(rare_action_types), 'action_type'] = 'Other'

categorial_cols = [
    'action_type', 'combined_shot_type', 'period', 'season', 'shot_type',
    'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'game_year',
    'game_month', 'opponent', 'loc_x', 'loc_y']

for cc in categorial_cols:
    dummies = pd.get_dummies(data_scrubbed[cc])
    dummies = dummies.add_prefix("{}#".format(cc))
    data_scrubbed.drop(cc, axis=1, inplace=True)
    data_scrubbed = data_scrubbed.join(dummies)

# Separate dataset for validation
unknown_mask = data['shot_made_flag'].isnull()
data_submit = data_scrubbed[unknown_mask]

# Separate dataset for training
X = data_scrubbed[~unknown_mask]
Y = target[~unknown_mask]


threshold = 0.90
vt = VarianceThreshold().fit(X)

# Find feature names
feat_var_threshold = data_scrubbed.columns[vt.variances_ > threshold * (1-threshold)]

#Random Forest
model = RandomForestClassifier()
model.fit(X, Y)

feature_imp = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["importance"])
feat_imp_20 = feature_imp.sort_values("importance", ascending=False).head(20).index
print "Random Forest"
print feat_imp_20
# Chi
X_minmax = MinMaxScaler(feature_range=(0,1)).fit_transform(X)
X_scored = SelectKBest(score_func=chi2, k='all').fit(X_minmax, Y)
feature_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': X_scored.scores_
    })

feat_scored_20 = feature_scoring.sort_values('score', ascending=False).head(20)['feature'].values
feat_scored_20
print "Chi"
print feat_imp_20
# Recursive Feature
rfe = RFE(LogisticRegression(), 20)
rfe.fit(X, Y)

feature_rfe_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': rfe.ranking_
    })

feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
print "recursive"
print feat_rfe_20

features = np.hstack([
        feat_var_threshold, 
        feat_imp_20,
        feat_scored_20,
        feat_rfe_20
    ])

features = np.unique(features)
print('Final features set:\n')
for f in features:
    print("\t-{}".format(f))
#Prepare Data
data_scrubbed = data_scrubbed.ix[:, features]
data_submit = data_submit.ix[:, features]
X = X.ix[:, features]

print('Clean dataset shape: {}'.format(data_scrubbed.shape))
print('Subbmitable dataset shape: {}'.format(data_submit.shape))
print('Train features shape: {}'.format(X.shape))
print('Target label shape: {}'. format(Y.shape))

# Evaluate Algos
seed = 7
processors=1
num_folds=3
num_instances=len(X)
scoring='log_loss'

kfold = KFold(n=num_instances, n_folds=num_folds, random_state=seed)

# Prepare some basic models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('K-NN', KNeighborsClassifier(n_neighbors=5)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVC', SVC(probability=True)))

# Evaluate each model in turn
results = []
names = []

for name, model in models:
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
    results.append(cv_results)
    names.append(name)
    print("{0}: ({1:.3f}) +/- ({2:.3f})".format(name, cv_results.mean(), cv_results.std()))
###Ensembles
# Bagged Decision    
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))
# Random Forest
num_trees = 100
num_features = 10
model = RandomForestClassifier(n_estimators=num_trees, max_features=num_features)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))
# extra trees
num_trees = 100
num_features = 10
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=num_features)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))
# ada boost
model = AdaBoostClassifier(n_estimators=100, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))
# stocastic gradient
model = GradientBoostingClassifier(n_estimators=100, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))
# logistic
lr_grid = GridSearchCV(
    estimator = LogisticRegression(random_state=seed),
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 1, 10, 100, 1000]
    }, 
    cv = kfold, 
    scoring = scoring, 
    n_jobs = processors)
lr_grid.fit(X, Y)
print "LR"
print(lr_grid.best_score_)
print(lr_grid.best_params_)
# linear discriminant analysis
lda_grid = GridSearchCV(
    estimator = LinearDiscriminantAnalysis(),
    param_grid = {
        'solver': ['lsqr'],
        'shrinkage': [0, 0.25, 0.5, 0.75, 1],
        'n_components': [None, 2, 5, 10]
    }, 
    cv = kfold, 
    scoring = scoring, 
    n_jobs = processors)
lda_grid.fit(X, Y)
print "LDA"
print(lda_grid.best_score_)
print(lda_grid.best_params_)
#k-nn
knn_grid = GridSearchCV(
    estimator = Pipeline([
        ('min_max_scaler', MinMaxScaler()),
        ('knn', KNeighborsClassifier())
    ]),
    param_grid = {
        'knn__n_neighbors': [25],
        'knn__algorithm': ['ball_tree'],
        'knn__leaf_size': [2, 3, 4],
        'knn__p': [1]
    }, 
    cv = kfold, 
    scoring = scoring, 
    n_jobs = processors)
knn_grid.fit(X, Y)
print "KNN"
print(knn_grid.best_score_)
print(knn_grid.best_params_)
# random forest
rf_grid = GridSearchCV(
    estimator = RandomForestClassifier(warm_start=True, random_state=seed),
    param_grid = {
        'n_estimators': [100, 200],
        'criterion': ['gini', 'entropy'],
        'max_features': [18, 20],
        'max_depth': [8, 10],
        'bootstrap': [True]
    }, 
    cv = kfold, 
    scoring = scoring, 
    n_jobs = processors)
rf_grid.fit(X, Y)
print "Random Forest"
print(rf_grid.best_score_)
print(rf_grid.best_params_)
# Ada Boost
ada_grid = GridSearchCV(
    estimator = AdaBoostClassifier(random_state=seed),
    param_grid = {
        'algorithm': ['SAMME', 'SAMME.R'],
        'n_estimators': [10, 25, 50],
        'learning_rate': [1e-3, 1e-2, 1e-1]
    }, 
    cv = kfold, 
    scoring = scoring, 
    n_jobs = processors)
ada_grid.fit(X, Y)
print "Ada"
print(ada_grid.best_score_)
print(ada_grid.best_params_)
#Gradient
gbm_grid = GridSearchCV(
    estimator = GradientBoostingClassifier(warm_start=True, random_state=seed),
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [2, 3, 4],
        'max_features': [10, 15, 20],
        'learning_rate': [1e-1, 1]
    }, 
    cv = kfold, 
    scoring = scoring, 
    n_jobs = processors)
gbm_grid.fit(X, Y)
print "Gradient"
print(gbm_grid.best_score_)
print(gbm_grid.best_params_)
# Voting
# Create sub models
estimators = []
estimators.append(('lr', LogisticRegression(penalty='l2', C=1)))
estimators.append(('gbm', GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, max_features=15, warm_start=True, random_state=seed)))
estimators.append(('rf', RandomForestClassifier(bootstrap=True, max_depth=8, n_estimators=200, max_features=20, criterion='entropy', random_state=seed)))
estimators.append(('ada', AdaBoostClassifier(algorithm='SAMME.R', learning_rate=1e-2, n_estimators=10, random_state=seed)))
# create the ensemble model
ensemble = VotingClassifier(estimators, voting='soft', weights=[2,3,3,1])
results = cross_val_score(ensemble, X, Y, cv=kfold, scoring=scoring,n_jobs=processors)
print "Voting"
print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))














