from src.logger import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
import joblib
import os
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


class CustomerClustering:
    def __init__(self, data, model_path='best_model.joblib'):
        self.data = data
        self.model_path = model_path
        self.model = None
        self.features = ['Recency', 'Frequency', 'MonetaryValue']
        self.X = self.data[self.features]
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def cluster_customers(self):
        logging.info("Starting clustering...")
        
        # K-means model pipeline
        kmeans_pipeline = Pipeline([
            ('kmeans', MiniBatchKMeans(n_clusters=3, random_state=42))  # Adjust the number of clusters as needed
        ])
        
        # Fit the model
        kmeans_pipeline.fit(self.X_scaled)
        self.model = kmeans_pipeline
        
        # Evaluate the clusters with silhouette score
        silhouette_avg = silhouette_score(self.X_scaled, self.model.named_steps['kmeans'].labels_)
        logging.info(f'Silhouette Score: {silhouette_avg:.2f}')
        
        logging.info("Clustering completed.")



        
    def save_model(self):
        logging.info(f"Saving model to {self.model_path}...")
        joblib.dump(self.model, self.model_path)
        logging.info("Model saved.")
        
    def load_model(self):
        if os.path.exists(self.model_path):
            logging.info(f"Loading model from {self.model_path}...")
            self.model = joblib.load(self.model_path)
            logging.info("Model loaded.")
        else:
            logging.error(f"No model found at {self.model_path}.")
            raise NotFittedError("Model not found!")

    def execute_pipeline(self):
        if not os.path.exists(self.model_path):
            self.cluster_customers()
            self.save_model()
        else:
            self.load_model()

if __name__ == "__main__":
    preprocessed_data = pd.read_csv('data/engineered_data.csv')
    clustering_instance = CustomerClustering(preprocessed_data)
    clustering_instance.execute_pipeline()
    trained_model = clustering_instance.model
