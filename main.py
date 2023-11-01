from src.data_preprocessing import DataPreprocessor, preprocess_data
from src.evaluation import ClusterEvaluator
from src.modelling import CustomerClustering
import pandas as pd
import os

def main():
    # Paths
    raw_data_filepath = 'data/raw_data.csv'
    preprocessed_data_filepath = 'data/preprocessed_data.csv'
    engineered_data_filepath = 'data/engineered_data.csv'
    
    # Data Preprocessing
    preprocess_data(raw_data_filepath, preprocessed_data_filepath)
    
    # Load Preprocessed Data
    preprocessed_data = pd.read_csv(preprocessed_data_filepath)
    
    # Feature Engineering
    from src.feature_engineering import feature_engineering_pipeline
    engineered_data = feature_engineering_pipeline(preprocessed_data)
    
    # Save Engineered Data
    if engineered_data is not None:
        engineered_data.to_csv(engineered_data_filepath, index=False)
    
    # Clustering
    clustering_instance = CustomerClustering(data=engineered_data)
    clustering_instance.execute_pipeline()
    
    # Evaluation
    evaluator = ClusterEvaluator(data=engineered_data)
    evaluator.silhouette_evaluation()
    evaluator.visualize_clusters()

if __name__ == "__main__":
    main()
