import pandas as pd
from feature_engineering import feature_engineering_pipeline

preprocessed_data = pd.read_csv('data/preprocessed_data.csv')

engineered_data = feature_engineering_pipeline(preprocessed_data)

if engineered_data is not None:
    print(engineered_data.head(20))
else:
    print("Feature engineering pipeline did not complete successfully.")

