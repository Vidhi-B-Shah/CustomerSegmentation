import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.logger import logging
import os


class DataPreprocessor:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath, encoding='iso-8859-1')
        self.scaler = StandardScaler()

    def handle_missing_values(self):
        logging.info('Handling missing values...')
        self.data.dropna(subset=['Description'], inplace=True)
        self.data['CustomerID'].fillna(-1, inplace=True)

    def convert_datatypes(self):
        logging.info('Converting data types...')
        self.data['InvoiceDate'] = pd.to_datetime(self.data['InvoiceDate'])
        self.data['CustomerID'] = self.data['CustomerID'].astype('str')

    def remove_outliers(self):
        logging.info('Removing outliers...')
        self.data = self.data[(self.data['Quantity'] >= 1) & (self.data['UnitPrice'] >= 0)]

    def handle_duplicates(self):
        logging.info('Handling duplicate entries...')
        self.data.drop_duplicates(inplace=True)

    def feature_engineering(self):
        logging.info('Performing feature engineering...')
        self.data['TotalPrice'] = self.data['Quantity'] * self.data['UnitPrice']
        self.data['InvoiceYear'] = self.data['InvoiceDate'].dt.year
        self.data['InvoiceMonth'] = self.data['InvoiceDate'].dt.month
        self.data['InvoiceDay'] = self.data['InvoiceDate'].dt.day
        self.data['InvoiceHour'] = self.data['InvoiceDate'].dt.hour

    def scale_features(self):
        logging.info('Scaling features...')
        self.data[['Quantity', 'UnitPrice', 'TotalPrice']] = self.scaler.fit_transform(
            self.data[['Quantity', 'UnitPrice', 'TotalPrice']])

    def preprocess(self):
        logging.info('Starting data preprocessing...')
        self.handle_missing_values()
        self.convert_datatypes()
        self.handle_duplicates()
        self.remove_outliers()  # Moved this line up
        self.feature_engineering()
        self.scale_features()  # This should come after removing outliers
        logging.info('Data preprocessing completed.')
        return self.data


def preprocess_data(filepath, output_filepath):
    preprocessor = DataPreprocessor(filepath)
    preprocessed_data = preprocessor.preprocess()
    preprocessed_data.to_csv(output_filepath, index=False)
    logging.info(f'Preprocessed data saved to {output_filepath}')


filepath = 'data/raw_data.csv'
output_filepath = 'data/preprocessed_data.csv'
preprocess_data(filepath, output_filepath)
