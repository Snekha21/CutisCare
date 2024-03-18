import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn import metrics
import plotly.subplots as sp
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio

import warnings
warnings.filterwarnings("ignore") 
genre = st.radio(
    "Algorithm",
    ('K-Means', 'Agglometric clustering', 'DBSCAN','PCA'))



df = pd.read_csv("survey.csv")
st.subheader("DataFrame")
st.dataframe(df)

df = df.drop(index=138)
df = df.drop(index=139)
df = df.drop(index=140)
df = df.drop(index=141)
df = df.drop(index=142)
df = df.replace('Never', 0)


df = df.replace('Sometimes', 1)
df = df.replace('Always', 2)
df = df.replace('Female', 1)
df = df.replace('Male', 0)
df = df.replace('Prefer not to say', 0)
df = df.drop('Unnamed: 24', axis=1)
df = df.drop('Unnamed: 25', axis=1)
df = df.drop('Unnamed: 26', axis=1)
df = df.drop('Unnamed: 27', axis=1)

st.subheader("Cleaned data")
st.dataframe(df)
df = df.drop('Name', axis=1)

dfs=df


import matplotlib.pyplot as plt
import seaborn as sns
if(genre=="K-Means"):
# Define the subplot layout
    rows = 9
    cols = 2

    # Create a figure and axis objects using subplots()
    fig, axs = plt.subplots(rows, cols, figsize=(20, 30))

    # Plot the KDEs for each numeric column
    num_cols = [col for col in df.columns if df[col].dtype == 'int64']
    row = 0
    col = 0

    for col_name in num_cols:
        # Plot the KDE using seaborn's kdeplot()
        sns.kdeplot(df[col_name], ax=axs[row][col], shade=True)
        
        # Set the axis labels
        axs[row][col].set_xlabel(col_name)
        axs[row][col].set_ylabel('Density')
        
        # Increment row and column counters
        col += 1
        if col == cols:
            col = 0
            row += 1

    # Set the main title and adjust subplot spacing
    fig.suptitle('KDE Plot for Numeric Columns', fontsize=24)
    fig.tight_layout(pad=3.0)

    # Display the figure
    st.pyplot()


    plt.figure(figsize=(10,60))
    for i in range(2,17):
        plt.subplot(17,1,i+1)
        sns.distplot(df[df.columns[i]],kde_kws={'color':'b','bw': 0.1,'lw':3,'label':'KDE'},hist_kws={'color':'g'})
        plt.title(df.columns[i])
    plt.tight_layout()

    # Display the plot
    st.pyplot()


    sns.histplot(df['Y/N'], kde=True, stat='density', bins=20, color='#008000')

    # Create subplots for the remaining columns
    # fig, axs = plt.subplots(22, figsize=(10, 60))
    # for i in range(2,24):
    #     sns.histplot(ax=axs[i-2], data=df[df.columns[i]], kde=True, stat='density', bins=20, color='#008000')
    #     axs[i-2].set_title(df.columns[i])

    # # Add title and adjust spacing
    # fig.suptitle('Distribution Plots', fontsize=16)
    # fig.tight_layout()

    # # Display the plot
    # st.pyplot()


    fig = go.Figure(data=go.Heatmap(
            z=df.corr(),
            x=df.columns,
            y=df.columns,
            colorscale='Viridis',
            colorbar=dict(title='Correlation'),
            zmin=-1,
            zmax=1,
            showscale=True,
            hoverongaps=False,
            connectgaps=False,
            hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>Correlation: %{z:.2f}<extra></extra>'
        ))

    fig.update_layout(title='Correlation Heatmap', width=800, height=800)

    st.pyplot()




    scaled_df = scalar.fit_transform(df)
    from sklearn.cluster import KMeans
    kmeans = KMeans(3, random_state=0)
    labels = kmeans.fit(scaled_df).predict(scaled_df)
    plt.scatter(df['Age'], df['Y/N'], c=labels, s=40, cmap='viridis');
if(genre=='PCA'):
    scaled_df = scalar.fit_transform(df)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_df)
    pca_df = pd.DataFrame(data=principal_components ,columns=["PCA1","PCA2"])
    st.dataframe(pca_df)

    # inertia = []
    # range_val = range(1, 15)
    # for i in range_val:
    #     kmean = KMeans(n_clusters=i)
    #     kmean.fit_predict(pd.DataFrame(scaled_df))
    #     inertia.append(kmean.inertia_)

    # data = [go.Scatter(x=list(range_val), y=inertia, mode='lines+markers')]
    # layout = go.Layout(title='The Elbow Method using Inertia',
    #                 xaxis=dict(title='Values of K'),
    #                 yaxis=dict(title='Inertia'))

    # fig = go.Figure(data=data, layout=layout)

    # # st.pyplot(fig)
    # st.pyplot()

    kmeans_model=KMeans(4)
    kmeans_model.fit_predict(scaled_df)
    pca_df_kmeans= pd.concat([pca_df,pd.DataFrame({'cluster':kmeans_model.labels_})],axis=1)

    colors = ['red', 'green', 'blue', 'black']

    # Create a scatter plot for each cluster
    for i in range(len(pca_df_kmeans['cluster'].unique())):
        df = pca_df_kmeans[pca_df_kmeans['cluster'] == i]
        plt.scatter(df['PCA1'], df['PCA2'], c=colors[i], label='Cluster {}'.format(i), alpha=0.5)

    # Add title and axes labels
    plt.title('Clustering using K-Means Algorithm')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')

    # Add legend and adjust spacing
    plt.legend()
    plt.tight_layout()

    # Display the plot
    st.pyplot()


    cluster_df = pd.concat([df,pd.DataFrame({'Cluster':kmeans_model.labels_})],axis=1)
    # st.dataframe(cluster_df)

    # plt.bar(cluster_df['Cluster'], color=cluster_df['Cluster'])

    # # Add title and axes labels
    # plt.title('Countplot using Plotly')
    # plt.xlabel('Cluster')
    # plt.ylabel('Count')

    # # Display the plot
    # st.pyplot()

    nrows = 1
    ncols = cluster_df['Cluster'].nunique()

    # Create a figure with subplots using plt.subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5))

    # Loop through each variable and each cluster, and create a histogram for each combination
    for i, c in enumerate(cluster_df.drop(['Cluster'], axis=1)):
        for j, cl in enumerate(cluster_df['Cluster'].unique()):
            # Select data for the current variable and cluster
            data = cluster_df[c][cluster_df['Cluster'] == cl].dropna()
            # Create a histogram using plt.hist
            axes[j].hist(data, bins=20, alpha=0.5, label='Cluster {}'.format(cl), color=f'C{j}')
            # Add a title and axes labels for the subplot
            axes[j].set_title('Cluster {}'.format(cl))
            axes[j].set_xlabel(c)
            axes[j].set_ylabel('Count')

    # Add a title for the entire figure
    fig.suptitle('Histograms by Cluster')

    # Adjust the layout of the subplots and add legends
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.legend()

    # Display the plot
    st.pyplot()

if(genre=="Agglometric clustering"):

    from sklearn.mixture import GaussianMixture
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    scalar=StandardScaler()

    import pandas as pd
    # data = pd.read_csv('.csv')

    # df = pd.read_csv('scaled_df.csv')
    df = scalar.fit_transform(df)
    # df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillnaPURCHASESdf["CREDIT_LIMIT"].mean())
    st.dataframe(df)



    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler, normalize
    from sklearn.metrics import silhouette_score
    import scipy.cluster.hierarchy as shc

    X_normalized = normalize(df)
    X_normalized = pd.DataFrame(X_normalized)


    pca = PCA(n_components = 2)
    X_principal = pca.fit_transform(X_normalized)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']


    plt.figure(figsize =(8, 8))
    plt.title('Visualising the data')
    Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward')))
    st.pyplot()


    ac2 = AgglomerativeClustering(n_clusters = 2)
    
    # Visualizing the clustering
    plt.figure(figsize =(3, 3))
    plt.scatter(X_principal['P1'], X_principal['P2'],
            c = ac2.fit_predict(X_principal), cmap ='rainbow')
    st.pyplot()

    ac3 = AgglomerativeClustering(n_clusters = 3)
    
    plt.figure(figsize =(6, 6))
    plt.scatter(X_principal['P1'], X_principal['P2'],
                c = ac3.fit_predict(X_principal), cmap ='rainbow')
    st.pyplot()


    ac4 = AgglomerativeClustering(n_clusters = 4)
    
    plt.figure(figsize =(6, 6))
    plt.scatter(X_principal['P1'], X_principal['P2'],
                c = ac4.fit_predict(X_principal), cmap ='rainbow')
    st.pyplot()

    ac5 = AgglomerativeClustering(n_clusters = 6)
    
    plt.figure(figsize =(6, 6))
    plt.scatter(X_principal['P1'], X_principal['P2'],
                c = ac5.fit_predict(X_principal), cmap ='rainbow')
    st.pyplot()


    ac6 = AgglomerativeClustering(n_clusters = 6)
    
    plt.figure(figsize =(6, 6))
    plt.scatter(X_principal['P1'], X_principal['P2'],
                c = ac6.fit_predict(X_principal), cmap ='rainbow')
    st.pyplot()


    k = [2, 3,4,5,6]
    
    # Appending the silhouette scores of the different models to the list
    silhouette_scores = []
    silhouette_scores.append(
            silhouette_score(X_principal, ac2.fit_predict(X_principal)))
    silhouette_scores.append(
            silhouette_score(X_principal, ac3.fit_predict(X_principal)))
    silhouette_scores.append(
            silhouette_score(X_principal, ac4.fit_predict(X_principal)))
    silhouette_scores.append(
            silhouette_score(X_principal, ac5.fit_predict(X_principal)))
    silhouette_scores.append(
            silhouette_score(X_principal, ac6.fit_predict(X_principal)))
    
    # Plotting a bar graph to compare the results
    plt.bar(k, silhouette_scores)
    plt.xlabel('Number of clusters', fontsize = 20)
    plt.ylabel('S(i)', fontsize = 20)
    st.pyplot()


if(genre=="DBSCAN"):
    scaled_df = scalar.fit_transform(df)
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_moons
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import v_measure_score
    st.subheader("DBSCAN")

    dbscan_cluster = DBSCAN(eps=3.5, min_samples=8)
    dbscan_cluster.fit(scaled_df)
    # 0.83,3
    # Visualizing DBSCAN
    plt.scatter(dfs['Age'], 
                dfs['Y/N'], 
                c=dbscan_cluster.labels_, 
                label=scaled_df)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    st.pyplot()

    # Number of Clusters
    labels=dbscan_cluster.labels_
    N_clus=len(set(labels))
    st.success('Estimated no. of clusters: %d' % N_clus)
    st.info(np.unique(labels))

    # Identify Noise
    n_noise = list(dbscan_cluster.labels_).count(-1)
    st.success('Estimated no. of noise points: %d' % n_noise)
    sc = metrics.silhouette_score(scaled_df, labels)

    st.success("dbscan: silhouttte:%f "%sc)
    # # Calculating v_measure
    # print('v_measure =', v_measure_score(df, labels))

