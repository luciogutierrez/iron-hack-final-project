# Imports
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.cluster as cluster
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

# Extraction
df = pd.read_csv('datasets/marvel-powers.csv')
df.head()

# Analysis and Transformation
df.info()
df.isna().sum()

# Transformation
# since we have 3 row with null we can droped without affect the dataset
df = df.dropna()

# selccionamos variables numericas para entrenar el modelo
n_cols = [col for col in df.columns.values if col not in ['Name', 'Alignment']]
n_cols

# Clustering
## Identifing number of clusters

#Elbow Method/ Método de codo nos funciona apra identificar número adecuado de posibles clúster
K = range(1,15)
sum_of_squared_distances = []
for k in K:
  model = KMeans(n_clusters=k).fit(df[n_cols])
  sum_of_squared_distances.append(model.inertia_)
plt.plot(K, sum_of_squared_distances,'wx-')
plt.xlabel('K Values')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method')
plt.show();

## KMean Clusterization
# la gráfica sugiere que 3 clusters es adecuado para el set de datos
kmeans = KMeans(n_clusters=3, random_state=111)
# entrenamos el modelo
df_clusters = kmeans.fit(df[n_cols])
df_clusters

# asignamos un grupo a cada observación y luego agregamos este dato al dataset
df['cluster'] = df_clusters.fit_predict(df[n_cols])
df.cluster.value_counts()
df[df.cluster == 0]
df.groupby(['cluster'])['Total'].mean()

g=sns.pairplot(data=df, vars=['Intelligence','Strength'], hue='Alignment', height=3, kind= 'scatter')
g.fig.set_size_inches(15,10);

## Hierarchical Clustering
# tomamos una muestra del dataset para generar un dendrograma claro y ordenado
df_sample = df[n_cols].sample(n=100)
hier_clust = AgglomerativeClustering(linkage='ward')
df_hier = hier_clust.fit(df_sample)

# función para plotter un dendrogram
def plot_dendrogram(model, **kwargs):
    children = model.children_
    distance = np.arange(children.shape[0])
    no_of_observations = np.arange(2, children.shape[0]+2)
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

# generamos el dendogram del modelo
plot_dendrogram(df_hier, labels=df_hier.labels_);

df[n_cols].head()

X = df[n_cols].to_numpy()
labels = kmeans.predict(X)
center = kmeans.cluster_centers_
colors = ["red","yellow","green","blue"]

asignar=[]
for row in labels:
  asignar.append(colors[row])

plt.figure(figsize=(15,10))
feature_1 = X[:,1]
feature_2 = X[:,2]
plt.scatter(feature_1, feature_2, c=asignar, alpha=.7)
plt.scatter(center[:,0], center[:,1], marker = "*", c='black', s=240)
plt.show;

plt.figure(figsize=(15,10))
plt.scatter(X[:,0],X[:,1]);

## Nearest Neighbor Clustering
# Clusterización con Vecinos Cercanos
nearest_neighbors = NearestNeighbors(n_neighbors = 5)
nearest_neighbors.fit(X)
distances, indices = nearest_neighbors.kneighbors(X)
distances= np.sort(distances, axis=0)[:,1]
plt.figure(figsize=(15,10))
plt.plot(distances)
plt.show()

# Min_samples = si tenemos más de 2 minesiones, min samples = 2*dim , en este caso sería de 4
dbscan_1 = DBSCAN(eps=.20, min_samples = 4)
dbscan_1.fit(X)
labels= dbscan_1.labels_
df['cluster_hdbscan'] = labels
df.cluster_hdbscan.value_counts()

labels_mask_general = np.zeros_like(labels,dtype=bool)
n_noise_ = list(labels).count(-1)
unique_labels = set(labels)
n_clusters_ = len (set(labels))- (1 if -1 in labels else 0 )

# Visualizacion distancia .15
plt.figure(figsize=(15,10))
for k, col in zip(unique_labels, colors):
  if k==-1:
    col = "k"

  clase = (labels ==k)  
  # Es para lo que NO es ruido
  xy = X[clase & labels_mask_general]
  plt.plot(xy[:,0],xy[:,1],"o", markerfacecolor=col, markersize = 18, label=k, alpha =1)
    # Es para lo que SI es ruido
  xy = X[clase & ~labels_mask_general]
  plt.plot(xy[:,0],xy[:,1],"^", markerfacecolor=col, markersize = 18, alpha =1)

plt.title("Numero estimado de clusters: %d" %n_clusters_)
plt.legend()
plt.show()

## Gaussian Misture
fig, ax = plt.subplots(figsize=(15, 10))

n_components = range(1, 40)
covariance_types = ['spherical', 'tied', 'diag', 'full']

for covariance_type in covariance_types:
    valores_bic = []
    
    for i in n_components:
        modelo = GMM(n_components=i, covariance_type=covariance_type, random_state=123)
        modelo = modelo.fit(X)
        valores_bic.append(modelo.bic(X))
        
    ax.plot(n_components, valores_bic, label=covariance_type)
ax.set_title("Valores BIC")
ax.set_xlabel("Número componentes")
ax.legend();

## GMM
x_frame = pd.DataFrame(X)
x_frame = x_frame[[0,1]]
plt.figure(figsize=(30,10))
x_frame.plot(kind='kde', figsize=(13,7))
plt.show()

gm =GMM( n_components = 5, covariance_type = 'full', random_state=123)
gm.fit(X)
labels= gm.predict(X)
frame = pd.DataFrame(X)
frame['cluster']=labels
frame.head()

color = ['blue','green','cyan','black','orange']
for k in range(0,5):
  data = frame[frame["cluster"]==k]
  plt.scatter(data[1], data[2],c=color[k])
plt.show()  

plt.scatter(X.T[0], X.T[1], color='g');

# General functión to plot clustering
def plot_clusters(data, algorithm, args, kwds):
  start_time= time.time()
  labels = algorithm(*args, **kwds).fit_predict(data)
  end_time= time.time()
  palette = sns.color_palette('deep',np.unique(labels).max()+1)
  colors = [palette[x] if x>=0 else (0.0,0.0,0.0) for x in labels]
  plt.scatter(data.T[0],data.T[1], c= colors)
  plt.title('Cluster encontrados por {}'.format(str(algorithm.__name__)),fontsize=24)
  plt.text(-.6,-25,'Clusterizacion en {:.2f} segs'.format(end_time - start_time), fontsize =14)

## Kmean Cluster
plot_clusters(X, cluster.KMeans, (), {'n_clusters':5,'random_state':123})

## DBScan Cluster
plot_clusters(X, cluster.DBSCAN, (), {'eps':7.2})

## GMM
plot_clusters(X, GMM, (), {'n_components':5,'covariance_type':'full','random_state':123})