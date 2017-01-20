import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model.logistic import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer


def load_data_from_database(local = False):
    """ Function which accesses the source database located, and loads the data in the table named 'dsi'.  The data is sorted in ascending order by the 'index' column of the table """

    if(local == False):
        connect_param = 'postgresql://dsi_student:correct horse battery staple@joshuacook.me:5432/dsi'
        engine = create_engine(connect_param)
        return pd.read_sql("SELECT * FROM madelon ORDER BY index ASC",
                           con=engine)

    else:
        return pd.read_csv("madelon_data.csv")

def make_data_dict(all_data, test_percent = 0.3, rand_seed = 742):
    """ Function to generate the initial data set into a set of features and labels, and further separates these into a training and test set.  It returns a dictionary containing the labels ['X_train', 'X_test','y_train', 'y_test'] """

    y = all_data['label']
    y = y.apply(lambda x:int(x > 0))
    X = all_data
    del X['index']
    del X['label']

    X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=test_percent,
                             random_state=rand_seed)

    return {'X_train':X_train, 'X_test':X_test,
            'y_train':y_train, 'y_test':y_test}

def general_transformer(transformer, data_dict):
    """ Function to transform the training and test sets.  Based on the flags, the training set features will be normalized with the StandardScaler and/or reduced based on the SelectKBest algorithms, then the test set features will have the same transformations applied """

    transformer.fit(data_dict['X_train'],data_dict['y_train'])
    data_dict['X_train'] = \
        pd.DataFrame(transformer.transform(data_dict['X_train']))
    data_dict['X_test'] = \
        pd.DataFrame(transformer.transform(data_dict['X_test']))

    data_dict.setdefault('transformers', []).append(transformer)
    
    return data_dict

def general_model(data_dict, model, test_scores=False,folds = 10):
    """ Function to test the data set using logistic regression with an array of parameters.  This function returns the input data_dict, with the keys 'metrics' and 'final model' assigned a data frame of metric results and the fit model on the entire training set after cross-validation is performed to produce the metrics. """
    
    scoring_types = ["accuracy","roc_auc","precision","recall","f1"]
    scores = []

    scoring_types = [("accuracy",accuracy_score),
                     ("roc_auc",roc_auc_score),
                     ("precision",precision_score),
                     ("recall",recall_score),
                     ("f1",f1_score)]

    scores = []
    for type in scoring_types:
        scores.append(
            {"Score":type[0],
             "Cross-Validation":\
             np.mean(cross_val_score(model,
                                     data_dict['X_train'],data_dict['y_train'],
                                     cv=folds,scoring=type[0]))})

        if test_scores:
            full_fit = model.fit(data_dict['X_train'],data_dict['y_train'])
            y_pred = full_fit.predict(data_dict['X_test'])
            test_score = type[1](data_dict['y_test'], y_pred)
            scores[len(scores) - 1]["Validation"] = test_score

        else:
            scores[len(scores) - 1]["Validation"] = \
                "Too Early for Validation"

    data_dict.setdefault('models', []).append( \
        model.fit(data_dict['X_train'],data_dict['y_train']))

    data_dict.setdefault('metrics', []).append( \
        pd.DataFrame(scores)[["Score","Cross-Validation","Validation"]])

    return data_dict
