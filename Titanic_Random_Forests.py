##########################
# TITANIC MACHINE LEARNING
##########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option ("display.max_columns", None)
pd.set_option ('display.expand_frame_repr', False)
pd.set_option ('display.width', 1000)

# Reading test and train datasets
train = pd.read_csv ("/Users/ozanguner/PycharmProjects/son_dsmlbc/datasets/titanic_train.csv")

test = pd.read_csv ("/Users/ozanguner/PycharmProjects/son_dsmlbc/datasets/titanic_test.csv")

train.isnull ().sum ()
test.isnull ().sum ()


def rare_encode_alt(dataframe, rare_ratio=0.01, threshold=10):
    rare_val_cols = [col for col in dataframe.columns if (dataframe[col].dtype == "O") & (
        (dataframe[col].value_counts () / len (dataframe) < rare_ratio).any ()) & (
                             dataframe[col].nunique () < threshold)]

    new_dataframe = dataframe.copy ()

    for i in rare_val_cols:
        tmp = new_dataframe[i].value_counts () / len (new_dataframe)
        rare_labels = tmp[tmp < rare_ratio].index
        new_dataframe[i] = np.where (new_dataframe[i].isin (rare_labels), 'Rare', new_dataframe[i])

    return new_dataframe


def titanic_prep(data):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Deleting Passenger Id variable
    dataframe = data.drop ("PassengerId", axis=1)

    # MISSING VALUES
    missing_cols = [col for col in test.columns if (test[col].isnull ().any ()) & (col != "Cabin")]
    for i in missing_cols:
        if i == "Age":
            dataframe[i].fillna (dataframe.groupby ("Pclass")[i].transform ("median"), inplace=True)
        elif dataframe[i].dtype == "O":
            dataframe[i].fillna (dataframe[i].mode ()[0], inplace=True)
        else:
            dataframe[i].fillna (dataframe[i].median (), inplace=True)

    # Examining "Cabin" if there is a cabin number or not and checking it that affect to "Survived" variable or not
    dataframe.loc[dataframe["Cabin"].notna (), "NEW_IsCabin"] = 1
    dataframe.loc[dataframe["Cabin"].isnull (), "NEW_IsCabin"] = 0

    # Deleting "Cabin" variable
    dataframe.drop ("Cabin", axis=1, inplace=True)

    # FEATURE ENGINEERING
    # Extracting title of passengers from "Name" Variable
    dataframe["NEW_TITLE"] = dataframe["Name"].str.extract ('([A-Za-z]+)\.', expand=False)

    # Aggregating rare values in "NEW_TITLE" variable
    new_df = rare_encode_alt (dataframe, 0.02, 20)

    # Creating "IsAlone" Feature
    new_df.loc[(new_df["Parch"] + new_df["SibSp"]) == 0, "NEW_isAlone"] = 1
    new_df.loc[(new_df["Parch"] + new_df["SibSp"]) > 0, "NEW_isAlone"] = 0
    new_df.drop (["Parch", "SibSp"], axis=1, inplace=True)

    # Passengers' Welfare Level
    new_df["NEW_AGE_CAT"] = pd.cut (new_df["Age"], bins=[0, 18, 35, 55, new_df["Age"].max ()],
                                    labels=[4, 3, 2, 1]).astype (int)
    new_df["NEW_PCLASS_SCORING"] = new_df["Pclass"].map ({1: 3, 2: 2, 3: 1})  # high value is better

    new_df["NEW_WELFARE_LEVEL"] = new_df["NEW_PCLASS_SCORING"] * new_df["NEW_AGE_CAT"] * new_df["Fare"]

    # Encoding the columns that have more than two classes
    multiclass_cat_cols = [col for col in new_df.columns if 25 > new_df[col].nunique () > 2]

    new_df = pd.get_dummies (data=new_df, columns=multiclass_cat_cols, drop_first=False)

    # Encoding the columns that have two classes
    new_df["NEW_isMALE"] = new_df["Sex"].map ({"male": 1, "female": 0})
    new_df = new_df[[col for col in new_df.columns if new_df[col].dtype != "O"]]

    return new_df


train_prep = titanic_prep (train)
train_prep.shape
train_prep.isnull ().sum ()

test_prep = titanic_prep (test)
test_prep.shape
test_prep.isnull ().sum ()

# Examining the outliers with Local Outlier Factor
from sklearn.neighbors import LocalOutlierFactor

# For Train set
lof = LocalOutlierFactor (n_neighbors=20)
pred = lof.fit_predict (train_prep)
train_scores = lof.negative_outlier_factor_

np.sort (train_scores)
threshold_value = np.sort (train_scores)[10]

train_prep = train_prep[train_scores >= threshold_value]

train_prep.head ()

# For Test set
lof = LocalOutlierFactor (n_neighbors=20)
pred = lof.fit_predict (test_prep)
test_scores = lof.negative_outlier_factor_

np.sort (test_scores)
threshold_test = np.sort (test_scores)[1]
supp_value = test_prep[test_scores == threshold_test]
supp_array = supp_value.to_records (index=False)

outlier_df = test_prep[test_scores < threshold_test]
outlier_array = outlier_df.to_records (index=False)

outlier_array[:] = supp_array

test_prep.loc[outlier_df.index] = pd.DataFrame (outlier_array, index=outlier_df.index)

test_prep.head ()
test_prep.shape

# MACHINE LEARNING
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

X = train_prep.drop (["Survived"], axis=1)
y = train_prep["Survived"]

X_train, X_test, y_train, y_test = train_test_split (X, y, random_state=17, test_size=0.2)

rfc = RandomForestClassifier (random_state=17)
rfc_model = rfc.fit (X_train, y_train)

y_pred = rfc_model.predict (X_test)
print (classification_report (y_test, y_pred))
# recall => 0.75
# precision => 0.80
# f1-score => 0.77

y_prob = rfc_model.predict_proba (X_test)[:, 1]
roc_auc_score (y_test, y_prob)
# AUC => 0.86114


# MODEL TUNING
rfc_model = RandomForestClassifier (random_state=17)

rfc_params = {"max_depth": [5, 8, 10, None],
              "max_features": [3, 5, 10, 20],
              "n_estimators": [100, 500, 1000, 2000],
              "min_samples_split": [2, 5, 8]}

rfc_cv = GridSearchCV (rfc_model, rfc_params, cv=5, n_jobs=-1, verbose=True).fit (X_train, y_train)

rfc_cv.best_params_

# FINAL MODEL
tuned_rfc = RandomForestClassifier (random_state=17, **rfc_cv.best_params_).fit (X_train, y_train)
tuned_pred = tuned_rfc.predict (X_test)
print (classification_report (y_test, tuned_pred))
# recall => 0.75
# precision => 0.84
# f1-score => 0.79

y_prob = tuned_rfc.predict_proba (X_test)[:, 1]
roc_auc_score (y_test, y_prob)


# AUC => 0.86566


# Feature Importances
def plot_importance(model, features, num=len (X), save=False):
    feature_imp = pd.DataFrame ({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure (figsize=(10, 10))
    sns.set (font_scale=1)
    sns.barplot (x="Value", y="Feature", data=feature_imp.sort_values (by="Value",
                                                                       ascending=False)[0:num])
    plt.title ('Features')
    plt.tight_layout ()
    plt.show ()
    if save:
        plt.savefig ('importances.png')


plot_importance (tuned_rfc, X_train)

# NEW_TITLE_Mr is the most important feature


# TEST PREDICTION

titanic_predicts = pd.DataFrame ()
titanic_predicts["PassengerId"] = test["PassengerId"]
titanic_predicts["Survived"] = tuned_rfc.predict (test_prep)

titanic_no_index = titanic_predicts.set_index ("PassengerId")

titanic_no_index.to_csv ("ml_predictions.csv")
