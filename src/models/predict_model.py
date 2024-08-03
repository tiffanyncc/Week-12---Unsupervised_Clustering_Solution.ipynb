class KMeansPredict:
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_cluster_centers(self):
        return self.model.cluster_centers_
    
    def evaluate_silhouette(self, X):
        from sklearn.metrics import silhouette_score
        labels = self.model.predict(X)
        return silhouette_score(X, labels)
