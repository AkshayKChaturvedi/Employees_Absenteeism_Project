from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from math import sqrt


def split_data(x, y, test_size=0.20, random_state=2, **kwargs):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, **kwargs)

    return x_train, x_test, y_train, y_test


def model_fitting_and_get_training_error(model, x_train, y_train, **kwargs):

    classifier = model(**kwargs)

    classifier.fit(x_train, y_train)

    predictions_train = classifier.predict(x_train)

    training_error = sqrt(mean_squared_error(y_train, predictions_train))

    return predictions_train, training_error, classifier


def get_predictions_and_test_error(x_test, classifier, y_test):

    predictions_test = classifier.predict(x_test)

    test_error = sqrt(mean_squared_error(y_test, predictions_test))

    return predictions_test, test_error


def outlier_removal(data, cols):
    for i in cols:
        print(i)
        q75, q25 = np.percentile(data.loc[:, i], [75, 25])
        iqr = q75-q25
        min_value = q25 - (1.5*iqr)
        max_value = q75 + (1.5*iqr)
        print(q75, q25, iqr, min_value, max_value)
        data = data.drop(data[data.loc[:, i] < min_value].index)
        data = data.drop(data[data.loc[:, i] > max_value].index)
        print(data.shape[0])
    return data


def impute_missing_values(data, col, method):
    if method == 'median':
        data[col] = round(data[col].fillna(data[col].median()))
        return data[col]
    if method == 'mode':
        data[col] = data[col].fillna(data[col].mode()[0])
        return data[col]
    else:
        return "Please pass 'median' or 'mode' as argument in method parameter"

# ----------------------------------------------------- End ------------------------------------------------------------
