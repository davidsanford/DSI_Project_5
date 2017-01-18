from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data_from_database():
    connect_param = 'postgresql://dsi_student:correct horse battery staple@joshuacook.me:5432/dsi'
    engine = create_engine(connect_param)
    return pd.read_sql("SELECT * FROM madelon", con=engine)

def make_data_dict(all_data, test_percent = 0.3, rand = 742):
    y = all_data['label']
    X = all_data
    del X['index']
    del X['label']
    X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=test_percent, random_state=rand)

    return {'X_train':X_train, 'X_test':X_test,
            'y_train':y_train, 'y_test':y_test}

def general_transformer():
    pass

def general_model():
    pass
