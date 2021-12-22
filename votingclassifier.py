from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


def votingclassifier():
    estimator = []
    estimator.append(('Lasso', LogisticRegression(C=1.242, penalty='l1', solver='liblinear', max_iter=1e7)))
    estimator.append(('Ridge', LogisticRegression(C=2.5, penalty='l2', solver='lbfgs', max_iter=1e7)))
    estimator.append(('Xgb', xgb.XGBClassifier(n_estimators=40, learning_rate=0.032, max_depth=3,
                                               min_child_weight=3.16, reg_alpha=0.1, reg_lambda=10, gamma=0.1,
                                               subsample=0.98, colsample_bytree=0.85, random_state=1)))
    estimator.append(('KNC', KNeighborsClassifier(algorithm='auto', n_neighbors=7, p=2, weights='distance')))
    estimator.append(('SVC_rbf', SVC(C=2, gamma=0.1, kernel='rbf', random_state=1, probability=True)))
    # estimator.append(('SVC_linear', SVC(C=1, kernel='linear', random_state=1, probability=True)))
    # estimator.append(('SVC_poly', SVC(C=0.1, gamma=0.1, kernel='poly', coef0=5, degree=3,  random_state=1, probability=True)))
    estimator.append(('Decision_tree', tree.DecisionTreeClassifier(criterion='entropy', max_depth=5,
                                                                    max_features=None, min_samples_split=5, random_state=1)))
    estimator.append(('random_forest', RandomForestClassifier(bootstrap=True, criterion='entropy', max_depth=10,
                                                            max_features='sqrt', min_samples_leaf=2, min_samples_split=2,
                                                            n_estimators=300, random_state=1)))
    voting = VotingClassifier(estimators=estimator, voting='soft')
    return voting


