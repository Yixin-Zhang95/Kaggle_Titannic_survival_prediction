import pandas as pd
import numpy as np
from preprocessing import process, min_max_scaler, family_survival, standardscaler
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from parameter_tunning import get_best_params_ridge, get_best_params_lasso, get_best_params_xgb, get_best_params_RF, \
    get_best_params_KNC, get_best_params_SVC_rbf, get_best_params_DT, get_best_params_SVC_poly, get_best_params_SVC_linear
from votingclassifier import votingclassifier
from outliers import outlier_detection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# read files
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# drop outliers in df_train
df_train = outlier_detection(df_train)

# extract "survived" in df_train
y_train = df_train['Survived']
# print(y_train.astype('object').value_counts())
PassengerId = df_test['PassengerId']
# df_train = df_train.drop(['PassengerId'], axis=1)
# df_test = df_test.drop(['PassengerId'], axis=1)
train_size = len(y_train)


# concat train and test
df = pd.concat([df_train, df_test], axis=0)
df.reset_index(inplace=True)
if 'index' in df.columns:
    df = df.drop('index', axis=1)


# label encoding, "FAMILY_survival"
df = family_survival(df)



# drop "Survived" in df
df = df.drop(['Survived', 'PassengerId'], axis=1)


# preprocessing
df = process(df)
train_features = df.columns
matrix = min_max_scaler(df)
# matrix = standardscaler(df)
x_train, x_test = matrix[:train_size, :], matrix[train_size:, :]

# cross validate
parameter = get_best_params_ridge(x_train, y_train, train_features)
parameter = get_best_params_lasso(x_train, y_train, train_features)
# best_estimator = get_best_params_xgb(x_train, y_train, train_features)
# best_estimator = get_best_params_DT(x_train, y_train, train_features)
# best_estimator = get_best_params_RF(x_train, y_train, train_features)
# best_estimator = get_best_params_KNC(x_train, y_train, train_features)
# best_estimator = get_best_params_SVC_rbf(x_train, y_train, train_features)
# best_estimator = get_best_params_SVC_linear(x_train, y_train, train_features)
# best_estimator = get_best_params_SVC_poly(x_train, y_train, train_features)


# predict in the test dataset
# xgb predictor
# model_xgb = xgb.XGBClassifier(n_estimators=40, learning_rate=0.032, max_depth=7, min_child_weight=3.16,
#                          reg_alpha=0.316, reg_lambda=0.316, gamma=1, subsample=0.98, colsample_bytree=0.85,
#                          random_state=1, verbosity=0)
# model_xgb = model_xgb.fit(x_train, y_train)
# predict_xgb = model_xgb.predict(x_test)


# # Lasso predictor
# model_lasso = LogisticRegression(C=1.54, penalty='l1', solver='liblinear', max_iter=1e7)
# model_lasso.fit(x_train, y_train)
# predict_lasso = model_lasso.predict_proba(x_test)
# # print(predict_lasso)
#
# # Ridge predictor
# model_ridge = LogisticRegression(C=2.9, penalty='l2', solver='lbfgs', max_iter=1e7)
# model_ridge.fit(x_train, y_train)
# predict_ridge = model_ridge.predict_proba(x_test)[:, 1]
# # print(predict_ridge)

# random_forest predictor
# model_rf = RandomForestClassifier(bootstrap=False, min_samples_leaf=3, n_estimators=50,
#               min_samples_split=10, max_features='sqrt', max_depth=6)
# model_rf.fit(x_train, y_train)
# predict_rf = model_rf.predict(x_test)



# predict = (predict_lasso + predict_ridge + predict_xgb) / 3

# cross_validate using voting classifier
# voting = votingclassifier()
# score = cross_val_score(voting, x_train, y_train, cv=5)
# print('cross_val_score', score)

#voting classifier
voting = votingclassifier()
voting = voting.fit(x_train, y_train)
predict = voting.predict(x_test)
#
#
#
# predict
df_predict = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predict.astype(int)})
df_predict.to_csv('stack'+'submit10.csv', encoding='utf8', index=False)



