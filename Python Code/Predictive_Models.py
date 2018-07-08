from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from Utility_Functions import split_data, model_fitting_and_get_training_error, get_predictions_and_test_error, \
    impute_missing_values

df = pd.read_excel('C:/Users/Dell/Downloads/Absenteeism_at_work.xls')

num = ['Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Transportation expense',
       'Hit target', 'Son', 'Pet', 'Weight', 'Height', 'Body mass index', 'Absenteeism time in hours']

cat = list(set(df.columns) - set(num))

# Fill missing values
for var in num:
    df[var] = impute_missing_values(data=df, col=var, method='median')

for var in cat:
    df[var] = impute_missing_values(data=df, col=var, method='mode')

# Get dummy variables for categorical variables
df = pd.get_dummies(data=df, columns=cat)

x_train, x_test, y_train, y_test = split_data(df.iloc[:, df.columns != 'Absenteeism time in hours'], df.iloc[:, -1])

# --------------------------------- Normalizing data with mean 0 and variance 1 ----------------------------------------

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# --------------------------------------------- Linear Regression ------------------------------------------------------

lr_predictions_train, lr_training_error, lr_classifier = model_fitting_and_get_training_error(LinearRegression,
                                                                                              x_train, y_train)

lr_predictions_test, lr_test_error = get_predictions_and_test_error(x_test, lr_classifier, y_test)

print('Training set error of Linear Regression is : ', lr_training_error)

print('Test set error of Linear Regression is : ', lr_test_error)

# -------------------------------------------- Support Vector Regression -----------------------------------------------

svr_predictions_train, svr_training_error, svr_classifier = model_fitting_and_get_training_error(SVR, x_train, y_train)

svr_predictions_test, svr_test_error = get_predictions_and_test_error(x_test, svr_classifier, y_test)

print('Training set error of Support Vector Regression is : ', svr_training_error)

print('Test set error of Support Vector Regression is : ', svr_test_error)

# -------------------------------------------- Gradient Boosted Regression ---------------------------------------------

gbr_predictions_train, gbr_training_error, gbr_classifier = \
    model_fitting_and_get_training_error(GradientBoostingRegressor, x_train, y_train, max_depth=2, random_state=0)

gbr_predictions_test, gbr_test_error = get_predictions_and_test_error(x_test, gbr_classifier, y_test)

print('Training set error of Gradient Boosted Tree is : ', gbr_training_error)

print('Test set error of Gradient Boosted Tree is : ', gbr_test_error)

# ------------------------------------------------ Decision Tree -------------------------------------------------------

dtr_predictions_train, dtr_training_error, dtr_classifier = \
    model_fitting_and_get_training_error(DecisionTreeRegressor, x_train, y_train, min_impurity_decrease=0.01,
                                         random_state=0)

dtr_predictions_test, dtr_test_error = get_predictions_and_test_error(x_test, dtr_classifier, y_test)

print('Training set error of Decision Tree is : ', dtr_training_error)

print('Test set error of Decision Tree is : ', dtr_test_error)

# ------------------------------------------------ Random Forest -------------------------------------------------------

rfr_predictions_train, rfr_training_error, rfr_classifier = \
    model_fitting_and_get_training_error(RandomForestRegressor, x_train, y_train, n_estimators=500, random_state=0)

rfr_predictions_test, rfr_test_error = get_predictions_and_test_error(x_test, rfr_classifier, y_test)

print('Training set error of Random Forest is : ', rfr_training_error)

print('Test set error of Random Forest is : ', rfr_test_error)

# ----------------------------------------------------- End ------------------------------------------------------------
