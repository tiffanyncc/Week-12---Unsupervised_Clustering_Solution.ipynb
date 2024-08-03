# Mall Customer Clustering

## Overview
This project aims to segment customers into different groups using an unsupervised clustering model. The Mall Customer dataset contains hypothetical customer data.

## Project Structure
- `data/`: Contains the dataset and data loading scripts.
  - `processed/`: The final, canonical data sets for modeling.
  - `raw/`: The original, immutable data dump.
- `features/`: Contains data preprocessing and feature engineering scripts.
- `models/`: Contains the machine learning models and cluster evaluation scripts.
  - `train_model.py`: Contains training functions for KMeans model.
  - `predict_model.py`: Contains prediction functions for KMeans model.
  - `kmeans_exercise.py`: Contains functions for the final kmeans exercise.
- `visualization/`: Contains data visualization scripts.
- `main.py`: Main script to run the project.

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

## Installation
1. Clone the repository.
2. Navigate to the project directory.
3. Install the required packages: `pip install -r requirements.txt`.

## Usage
1. Place the raw dataset (`mall_customers.csv`) in the `data/raw/` directory.
2. Run the training script: `python main.py`.
3. Check the output for the model evaluation results.
4. Visualizations and model plots will be displayed during the run.

## Logging
The project uses logging to track the training process. Logs are printed to the console.
