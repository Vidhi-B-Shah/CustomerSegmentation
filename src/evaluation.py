from src.logger import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
import joblib

class ClusterEvaluator:
    def __init__(self, model_path='best_model.joblib', data=None):
        self.model_path = model_path
        self.data = data
        self.features = ['Recency', 'Frequency', 'MonetaryValue']
        self.X = self.data[self.features]
        self.load_model()

    def load_model(self):
        logging.info(f'Loading model from {self.model_path}...')
        self.model = joblib.load(self.model_path)
        logging.info('Model loaded successfully.')

    def silhouette_evaluation(self):
        logging.info('Evaluating Silhouette Score...')
        assert len(self.model.named_steps['kmeans'].labels_) == self.X.shape[0], "Mismatch in number of samples."
        silhouette_avg = silhouette_score(self.X, self.model.named_steps['kmeans'].labels_)
        logging.info(f'Silhouette Score: {silhouette_avg:.2f}')
        
    def pca_reduction(self):
        logging.info('Performing PCA reduction...')
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.X)
        logging.info('PCA reduction completed.')
        return pca_result
        
    def visualize_clusters(self):
        logging.info('Visualizing clusters...')
        pca_result = self.pca_reduction()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=self.model.named_steps['kmeans'].labels_, palette='viridis', s=60)
        plt.title('Visualization of Clusters')
        plt.show()
        logging.info('Clusters visualized successfully.')

if __name__ == "__main__":
    logging.info('Starting the evaluation process...')
    preprocessed_data = pd.read_csv('data/engineered_data.csv')
    
    evaluator = ClusterEvaluator(data=preprocessed_data)
    evaluator.silhouette_evaluation()
    evaluator.visualize_clusters()
    
    logging.info('Evaluation process completed.')
