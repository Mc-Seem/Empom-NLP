from sklearn.cluster import KMeans
import pyLDAvis
from auxiliary.kmeans_to_pyLDAvis.kmeans_to_pyLDAvis import kmeans_to_prepared_data
import plotly.express as px
import pandas as pd


def kmeans_vis(data, kmeans: KMeans, feature_names: callable, filename="kmeans_vis.html"):
    """
    Produces an interactive html-file with visualization of clusters and their most valuable features using tweaked
    pyLDAvis.

    Parameters:
        data: (csr_matrix): sparse matrix of vectorized values
        kmeans: (KMeans): instance of a clustering algorithm to get the labels and cluster centroids from
        feature_names: (callable): function to retrieve the names for the features from vectorizer
        filename: (str): name for a final visualization file

    """

    prep = kmeans_to_prepared_data(data, feature_names, kmeans.cluster_centers_,
                                   kmeans.labels_, embedding_method='tsne')
    with open(filename, "w") as f:
        pyLDAvis.save_html(prep, f)


def visualise(df: pd.DataFrame, column: str, flag=False) -> None:
    """
    Takes dataframe, its column and shows histogram for this column.
    Depending on value of flag may show percentage values for sentiments correlating to the column.

    Parameters:
        df: (pd.DataFrame): data to visualize
        column: (str): name of a column, distribution of data from which is chosen to be visualized
        flag: (bool): whether or not to indicate percentage values for sentiments for column values
    """

    if flag:
        fig = px.histogram(df, x=column, color='sentiment', title=f'Тональность в столбце {column}', histnorm='percent')
        fig.show()
    else:
        fig = px.histogram(df, x=column, title=f'График для столбца {column}', histnorm='percent')
        fig.show()
