from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


def get_best_params_ridge(x_train, y_train, train_feature):
    # 2.90
    params = {'C': np.linspace(2.5, 3.5, 100), 'penalty': ['l2']}
    model = LogisticRegression(solver='lbfgs', max_iter=1e7)
    grid_model = GridSearchCV(model, param_grid=params, scoring='accuracy', n_jobs=4, cv=5,
                              verbose=1)
    grid_model.fit(x_train, y_train)
    score = grid_model.best_score_
    print('Optimal score:', score)
    print('Optimal parameters:', grid_model.best_params_)

    return grid_model.best_estimator_


def get_best_params_lasso(x_train, y_train, train_feature):
    # 1.54
    params = {'C': np.linspace(1, 3, 100), 'penalty': ['l1']}
    model = LogisticRegression(solver='liblinear')
    grid_model = GridSearchCV(model, param_grid=params, scoring='accuracy', n_jobs=4, cv=5,
                              verbose=1)
    grid_model.fit(x_train, y_train)
    score = grid_model.best_score_
    coef = grid_model.best_estimator_.coef_[0]
    d = {}
    for i in range(len(coef)):
        d[train_feature[i]] = coef[i]
    for feature, coef in sorted(d.items(), key=lambda x: abs(x[1]), reverse=True):
        print(feature, coef)

    print('Optimal score:', score)
    print('Optimal parameters:', grid_model.best_params_)

    return grid_model.best_estimator_


def get_best_params_xgb(x_train, y_train, train_feature):
    # n_estimators = 40, learning_rate = 0.032, max_depth = 3, min_child_weight = 3.16,
    #                          reg_alpha=3.25, reg_lambda=1, gamma=0.1, subsample=0.98, colsample_bytree=0.85,
    #                          random_state=1)

    # Optimal parameters: {'colsample_bytree': 0.85, 'gamma': 1.0, 'learning_rate': 0.032, 'max_depth': 7, 'min_child_weight': 3.16,
    # 'n_estimators': 40, 'random_state': 1, 'reg_alpha': 0.31622776601683794, 'reg_lambda': 0.31622776601683794, 'subsample': 0.98}
    params = {'n_estimators': [40],
              'learning_rate': [0.032],
              'max_depth': [3],
              'min_child_weight': [3.16],
              'reg_alpha': np.logspace(-1, 1, 5),
              'reg_lambda': np.logspace(-1, 1, 5),
              'gamma': np.logspace(-1, 1, 5),
              'subsample': [0.98],
              'colsample_bytree': [0.5, 0.85],
              'random_state': [1]}
    model = xgb.XGBClassifier(n_jobs=1, use_label_encoder=False, verbosity=0)
    grid_model = GridSearchCV(model, param_grid=params, scoring='accuracy', n_jobs=5, cv=5,
                              verbose=1, refit=False)
    grid_model.fit(x_train, y_train)
    score = grid_model.best_score_
    print('Optimal score:', score)
    print('Optimal parameters:', grid_model.best_params_)

    return None


def get_best_params_KNC(x_train, y_train, train_feature):
# 0.822 Optimal parameters: {'algorithm': 'auto', 'n_neighbors': 29, 'p': 1, 'weights': 'uniform'}
    params = {'n_neighbors': np.arange(3, 30, 2),
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto'],
              'p': [1, 2]}
    model = KNeighborsClassifier()
    grid_model = GridSearchCV(model, param_grid=params, scoring='accuracy', n_jobs=4, cv=5,
                              verbose=1)
    grid_model.fit(x_train, y_train)
    score = grid_model.best_score_
    print('Optimal score:', score)
    print('Optimal parameters:', grid_model.best_params_)

    return grid_model.best_estimator_


def get_best_params_SVC_rbf(x_train, y_train, train_feature):
# 0.833 Optimal parameters: {'C': 2, 'gamma': 0.1, 'kernel': 'rbf'}
    params = {'gamma': [0.01, 0.1, 0.5, 1, 2, 5],
              'C': [.1, 1, 2, 5],
              'kernel': ['rbf']}
    model = SVC()
    grid_model = GridSearchCV(model, param_grid=params, scoring='accuracy', n_jobs=4, cv=5,
                              verbose=1)
    grid_model.fit(x_train, y_train)
    score = grid_model.best_score_
    print('Optimal score:', score)
    print('Optimal parameters:', grid_model.best_params_)

    return grid_model.best_estimator_


def get_best_params_SVC_poly(x_train, y_train, train_feature):
# 0.833 Optimal parameters: {'C': 2, 'gamma': 0.1, 'kernel': 'rbf'}
    params = {'gamma': [0.01, 0.1, 0.5, 1, 2, 5],
              'C': [.1, 1, 2, 5],
              'coef0': np.linspace(0, 10, 5),
              'degree': [2, 3],
              'kernel': ['poly']}
    model = SVC()
    grid_model = GridSearchCV(model, param_grid=params, scoring='accuracy', n_jobs=4, cv=5,
                              verbose=1)
    grid_model.fit(x_train, y_train)
    score = grid_model.best_score_
    print('Optimal score:', score)
    print('Optimal parameters:', grid_model.best_params_)

    return grid_model.best_estimator_


def get_best_params_SVC_linear(x_train, y_train, train_feature):
# 0.833 Optimal parameters: {'C': 2, 'gamma': 0.1, 'kernel': 'rbf'}
    params = {'C': [.1, 1, 2, 5],
              'kernel': ['linear']}
    model = SVC()
    grid_model = GridSearchCV(model, param_grid=params, scoring='accuracy', n_jobs=4, cv=5,
                              verbose=1)
    grid_model.fit(x_train, y_train)
    score = grid_model.best_score_
    print('Optimal score:', score)
    print('Optimal parameters:', grid_model.best_params_)

    return grid_model.best_estimator_


def get_best_params_DT(x_train, y_train, train_feature):
# 0.8229 {'criterion': 'entropy', 'max_depth': 3, 'max_features': None, 'min_samples_split': 5}
    params = {'max_depth': [3, 5, 10, 20, 50],
              'criterion': ['entropy', 'gini'],
              'min_samples_split': [5, 10, 15, 30],
              'max_features': [None, 'auto', 'sqrt', 'log2']}
    model = tree.DecisionTreeClassifier()
    grid_model = GridSearchCV(model, param_grid=params, scoring='accuracy', n_jobs=4, cv=5,
                              verbose=1)
    grid_model.fit(x_train, y_train)
    score = grid_model.best_score_
    print('Optimal score:', score)
    print('Optimal parameters:', grid_model.best_params_)

    return grid_model.best_estimator_


def get_best_params_RF(x_train, y_train, train_feature):
# 0.833 Optimal parameters: {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 10,
# 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 300}
    params = {'n_estimators': [50, 150, 300, 450],
              'criterion': ['entropy'],
              'bootstrap': [True],
              'max_depth': [3, 5, 10],
              'max_features': ['auto','sqrt'],
              'min_samples_leaf': [2, 3],
              'min_samples_split': [2, 3]}
    model = RandomForestClassifier()
    grid_model = GridSearchCV(model, param_grid=params, scoring='accuracy', n_jobs=4, cv=5,
                              verbose=1)
    grid_model.fit(x_train, y_train)
    score = grid_model.best_score_
    print('Optimal score:', score)
    print('Optimal parameters:', grid_model.best_params_)

    return grid_model.best_estimator_