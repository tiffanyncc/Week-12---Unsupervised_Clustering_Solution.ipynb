from sklearn.cluster import KMeans
import pandas as pd

class KMeansTrain:
    def __init__(self, n_clusters=5, init='k-means++', n_init=10):
        self.model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init)
    
    def train(self, X):
        self.model.fit(X)
    
    def get_model(self):
        return self.model

def calculate_wss(df, feature_cols, k_range):
    wss_scores = []
    K = []
    for k in k_range:
        kmodel = KMeansTrain(n_clusters=k)
        kmodel.train(df[feature_cols])
        wss_scores.append(kmodel.model.inertia_)
        K.append(k)
    return pd.DataFrame({'cluster': K, 'WSS_Score': wss_scores})

def calculate_silhouette(df, feature_cols, k_range):
    from sklearn.metrics import silhouette_score
    silhouette_scores = []
    K = []
    for k in k_range:
        kmodel = KMeansTrain(n_clusters=k)
        kmodel.train(df[feature_cols])
        labels = kmodel.model.predict(df[feature_cols])
        silhouette_scores.append(silhouette_score(df[feature_cols], labels))
        K.append(k)
    return pd.DataFrame({'cluster': K, 'Silhouette_Score': silhouette_scores})
