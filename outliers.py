from collections import Counter
import numpy as np


def outlier_detection(df):
    features = ["Age", "SibSp", "Parch", "Fare"]
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1

        outlier_step = IQR * 1.5

        outlier_one_feature = list(df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index)
        outlier_indices.extend(outlier_one_feature)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v >= 3)

    # drop rows with outliers
    df = df.drop(multiple_outliers, axis=0).reset_index(drop=True)
    return df



