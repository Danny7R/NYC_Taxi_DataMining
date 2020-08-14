import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import urllib.request
import zipfile
import random
import itertools
import math
import shapefile
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
import matplotlib as mpl
import matplotlib.pyplot as plt
from funcs import *
import seaborn as sns
plt.style.use('ggplot')
# matplotlib inline


sf = shapefile.Reader("data/taxi_zones/taxi_zones.shp")
fields_name = [field[0] for field in sf.fields[1:]]
shp_dic = dict(zip(fields_name, list(range(len(fields_name)))))


zones = pd.read_csv('data/taxi_zones_xy.csv', header=0, sep=',')  # , delim_whitespace=True
zones = zones.drop(columns=['zone', 'borough'])
df = pd.read_csv('data/fhv_tripdata_2017-06.csv', header=0, sep=',')
df = df[['PULocationID', 'DOLocationID']]
df = df.mask(df > 263)
df = df.mask(df < 1)
df = df.dropna().sample(n=10000)

fhv_2019_06_DO = df.DOLocationID.to_frame()
fhv_2019_06_PU = df.PULocationID.to_frame()
print(fhv_2019_06_DO.shape, fhv_2019_06_PU.shape)
fhv_2019_06_DO = fhv_2019_06_DO.join(zones.set_index('LocationID'), on='DOLocationID')
fhv_2019_06_PU = fhv_2019_06_PU.join(zones.set_index('LocationID'), on='PULocationID')
print(fhv_2019_06_DO.shape, fhv_2019_06_PU.shape)
X_DO = fhv_2019_06_DO.drop(columns=['DOLocationID']).values
X_PU = fhv_2019_06_PU.drop(columns=['PULocationID']).values


std = StandardScaler()
X_DO = std.fit_transform(X_DO)
X_PU = std.transform(X_PU)
k_means_PU = KMeans(n_clusters=5).fit(X_PU)
k_means_DO = KMeans(n_clusters=5).fit(X_DO)
X_DO = std.inverse_transform(X_DO)
X_PU = std.inverse_transform(X_PU)

print("Silhouette Coefficient PU: %0.3f" % metrics.silhouette_score(X_PU, k_means_PU.labels_))
print("Silhouette Coefficient DO: %0.3f" % metrics.silhouette_score(X_DO, k_means_DO.labels_))

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
ax = plt.subplot(1, 2, 1)
ax.set_title("K-means - pickups")
# draw_zone_map(ax, sf, shp_dic, text=[' '])
plt.scatter(X_PU[:, 0], X_PU[:, 1], c=k_means_PU.labels_, cmap='rainbow', zorder=2)
centers_PU = std.inverse_transform(k_means_PU.cluster_centers_)
plt.scatter(centers_PU[:, 0], centers_PU[:, 1], color='black', zorder=3)

ax = plt.subplot(1, 2, 2)
ax.set_title("K-means - drop-offs")
# draw_zone_map(ax, sf, shp_dic, text=[' '])
plt.scatter(X_DO[:, 0], X_DO[:, 1], c=k_means_DO.labels_, cmap='rainbow', zorder=2)
centers_DO = std.inverse_transform(k_means_DO.cluster_centers_)
plt.scatter(centers_DO[:, 0], centers_DO[:, 1], color='black', zorder=3)

plt.show()
