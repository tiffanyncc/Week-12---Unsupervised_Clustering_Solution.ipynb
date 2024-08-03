import matplotlib.pyplot as plt
import seaborn as sns

def plot_pairplot(df):
    sns.pairplot(df[['Age','Annual_Income','Spending_Score']])
    plt.savefig('src/visualization/images/pairplot.png')
    plt.show()

def plot_clusters(df, x_col, y_col, cluster_col):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df, hue=cluster_col, palette='colorblind')
    plt.title(f'Clusters: {x_col} vs {y_col}')
    plt.savefig(f'src/visualization/images/clusters_{x_col}_vs_{y_col}.png')
    plt.show()

def plot_elbow(wss, title):
    plt.figure(figsize=(10, 6))
    plt.plot(wss['cluster'], wss['WSS_Score'], marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WSS Score')
    plt.title('Elbow Plot')
    plt.savefig(f'src/visualization/images/{title.replace(" ", "_").lower()}.png')
    plt.show()

def plot_silhouette(wss, title):
    plt.figure(figsize=(10, 6))
    plt.plot(wss['cluster'], wss['Silhouette_Score'], marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Plot')
    plt.savefig(f'src/visualization/images/{title.replace(" ", "_").lower()}.png')
    plt.show()