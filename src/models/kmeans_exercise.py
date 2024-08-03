import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def calculate_kmeans_exercise(df):
    k = range(3, 9)
    K = []
    WCSS = []
    for i in k:
        kmodel = KMeans(n_clusters=i, init='k-means++', n_init=10).fit(df[['Age', 'Annual_Income', 'Spending_Score']])
        wcss_score = kmodel.inertia_
        WCSS.append(wcss_score)
        K.append(i)
    wss = pd.DataFrame({'cluster': K, 'WSS_Score': WCSS})

    ss = []
    for i in k:
        kmodel = KMeans(n_clusters=i, init='k-means++', n_init=10).fit(df[['Age', 'Annual_Income', 'Spending_Score']])
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[['Age', 'Annual_Income', 'Spending_Score']], ypred)
        ss.append(sil_score)
    wss['Silhouette_Score'] = ss

    return wss
