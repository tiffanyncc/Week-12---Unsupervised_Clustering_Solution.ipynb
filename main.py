import os

# Set the environment variable to avoid the memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

import logging
import matplotlib.pyplot as plt
from src.data.make_dataset import load_data
from src.models.train_model import KMeansTrain, calculate_wss, calculate_silhouette
from src.models.predict_model import KMeansPredict
from src.models.kmeans_exercise import calculate_kmeans_exercise
from src.visualization.visualize import plot_pairplot, plot_clusters, plot_elbow, plot_silhouette

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def main():
    try:
        df = load_data('src/data/raw/mall_customers.csv')
        logging.info('Data loaded successfully.')
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        return

    try:
        plot_pairplot(df)
        logging.info('Pairplot displayed.')
    except Exception as e:
        logging.error(f'Error displaying pairplot: {e}')

    try:
        kmodel = KMeansTrain(n_clusters=5, n_init=10)
        kmodel.train(df[['Annual_Income', 'Spending_Score']])
        model = kmodel.get_model()
        logging.info('KMeans model trained.')
    except Exception as e:
        logging.error(f'Error training KMeans model: {e}')
        return

    try:
        predictor = KMeansPredict(model)
        df['Cluster'] = predictor.predict(df[['Annual_Income', 'Spending_Score']])
        logging.info('Clusters predicted and assigned.')
    except Exception as e:
        logging.error(f'Error predicting clusters: {e}')
        return

    try:
        plot_clusters(df, 'Annual_Income', 'Spending_Score', 'Cluster')
    except Exception as e:
        logging.error(f'Error plotting clusters: {e}')

    try:
        k_ranges = {
            'Annual_Income and Spending_Score': ['Annual_Income', 'Spending_Score'],
            'Age, Annual_Income and Spending_Score': ['Age', 'Annual_Income', 'Spending_Score']
        }
        for description, features in k_ranges.items():
            k_range = range(3, 9)
            wss_df = calculate_wss(df, features, k_range)
            silhouette_df = calculate_silhouette(df, features, k_range)
            combined_df = wss_df.merge(silhouette_df, on='cluster')
            logging.info(f'Elbow and silhouette scores calculated for {description}.')

            plot_elbow(combined_df, f'Elbow Plot for {description}')
            plot_silhouette(combined_df, f'Silhouette Plot for {description}')

    except Exception as e:
        logging.error(f'Error calculating or displaying elbow and silhouette scores: {e}')

    try:
        wss = calculate_kmeans_exercise(df)
        plot_elbow(wss, 'Elbow Plot for Age, Annual_Income, Spending_Score with KMeans++')
        plot_silhouette(wss, 'Silhouette Plot for Age, Annual_Income, Spending_Score with KMeans++')
        logging.info('KMeans++ exercise completed and plots displayed and saved.')
    except Exception as e:
        logging.error(f'Error during KMeans++ exercise: {e}')

if __name__ == '__main__':
    main()