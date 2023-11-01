import pandas as pd
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder


def compute_rfm(data):
    logging.info('Computing RFM features...')
    try:
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
        
        # Recency
        most_recent_purchase = data.groupby('CustomerID')['InvoiceDate'].max()
        recency = (data['InvoiceDate'].max() - most_recent_purchase).dt.days
        
        # Frequency
        frequency = data.groupby('CustomerID')['InvoiceNo'].nunique()
        
        # Monetary Value
        monetary_value = data.groupby('CustomerID')['TotalSpend'].sum()
        
        rfm_df = pd.DataFrame({'Recency': recency, 'Frequency': frequency, 'MonetaryValue': monetary_value})
        return rfm_df
    except Exception as e:
        logging.error(f'Error computing RFM features: {e}')
        return None

def additional_features(data):
    logging.info('Computing additional features...')
    try:
        data['TotalInvoiceSpend'] = data.groupby('InvoiceNo')['TotalSpend'].transform('sum')
        data['AvgPricePerItem'] = data['TotalSpend'] / data['Quantity']
        
        return data
    except Exception as e:
        logging.error(f'Error computing additional features: {e}')
        return None

def behavioral_features(data):
    logging.info('Computing behavioral features...')
    try:
        # Favorite Shopping Day
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
        data['DayOfWeek'] = data['InvoiceDate'].dt.day_name()
        fav_shopping_day = data.groupby('CustomerID')['DayOfWeek'].agg(lambda x: x.value_counts().index[0])
        
        # Average Days Between Purchases
        days_between_purchases = data.groupby('CustomerID')['InvoiceDate'].apply(lambda x: x.diff().mean())
        avg_days_between_purchases = days_between_purchases.dt.days
        
        behavioral_df = pd.DataFrame({'FavoriteShoppingDay': fav_shopping_day, 'AvgDaysBetweenPurchases': avg_days_between_purchases})
        return behavioral_df
    except Exception as e:
        logging.error(f'Error computing behavioral features: {e}')
        return None

def cancellation_insights(data):
    logging.info('Computing cancellation insights...')
    try:
        data['Cancelled'] = data['InvoiceNo'].str.contains('C', regex=False).fillna(0).astype(int)
        cancellation_rate = data.groupby('CustomerID')['Cancelled'].mean()
        cancellation_df = pd.DataFrame({'CancellationRate': cancellation_rate})
        return cancellation_df
    except Exception as e:
        logging.error(f'Error computing cancellation insights: {e}')
        return None


def geographic_features(data):
    logging.info('Encoding geographic features...')
    try:
        encoder = OneHotEncoder(drop='first', sparse=False)
        encoded_features = encoder.fit_transform(data[['Country']])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Country']))
        data = data.reset_index(drop=True)
        data_encoded = pd.concat([data, encoded_df], axis=1)
        return data_encoded
    except Exception as e:
        logging.error(f'Error encoding geographic features: {e}')
        return None

def feature_engineering_pipeline(data):
    logging.info('Starting the feature engineering pipeline...')
    try:
        data['TotalSpend'] = data['Quantity'] * data['UnitPrice']
        
        rfm_df = compute_rfm(data)
        data = additional_features(data)
        behavioral_df = behavioral_features(data)
        cancellation_df = cancellation_insights(data)
        data_encoded = geographic_features(data)
        
        # Merge all engineered features with the original data
        data_engineered = data_encoded.merge(rfm_df, on='CustomerID').merge(
            behavioral_df, on='CustomerID').merge(cancellation_df, on='CustomerID')

        logging.info('Feature engineering pipeline completed successfully.')
        return data_engineered
    except Exception as e:
        logging.error(f'Error in feature engineering pipeline: {e}')
        return None

preprocessed_data = pd.read_csv('data/preprocessed_data.csv')


engineered_data = feature_engineering_pipeline(preprocessed_data)

if engineered_data is not None:
    engineered_data.to_csv('data/engineered_data.csv', index=False)
    logging.info('Engineered data saved to data/engineered_data.csv')
