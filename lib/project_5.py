def load_data_from_database():
    from sqlalchemy import create_engine
    import pandas as pd
    connect_param = 'postgresql://dsi_student:correct horse battery staple@joshuacook.me:5432/dsi'
    engine = create_engine(connect_param)
    return pd.read_sql("SELECT * FROM madelon", con=engine)

def make_data_dict():
    pass

def general_transformer():
    pass

def general_model():
    pass
