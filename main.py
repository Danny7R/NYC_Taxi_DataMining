import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import shapefile
import matplotlib.pyplot as plt
from funcs import *
import seaborn as sns
plt.style.use('ggplot')
# matplotlib inline

#######################################################################
# from sqlalchemy import create_engine
# nyc_database = create_engine('sqlite:///nyc_database.db_DO')

# j, chunksize = 1, 100000
# for month in range(1,7):
#     fp = "nyc.2017-{0:0=2d}.csv".format(month)
#     for df in pd.read_csv(fp, chunksize=chunksize, iterator=True):
#         df = df.rename(columns={c: c.replace(' ', '_') for c in df.columns})
#         df['pickup_hour'] = [x[11:13] for x in df['tpep_pickup_datetime']]
#         df['dropoff_hour'] = [x[11:13] for x in df['tpep_dropoff_datetime']]
#         df.index += j
#         df.to_sql('table_record', nyc_database, if_exists='append')
#         j = df.index[-1] + 1
# del df
#######################################################################

sf = shapefile.Reader("data/taxi_zones/taxi_zones.shp")
fields_name = [field[0] for field in sf.fields[1:]]
shp_dic = dict(zip(fields_name, list(range(len(fields_name)))))

zones = pd.read_csv('data/taxi_zones_xy.csv', header=0, sep=',')  # , delim_whitespace=True
zones = zones.drop(columns=['zone', 'borough'])
df = pd.read_csv('data/fhv_tripdata_2019-06.csv', header=0, sep=',')
df = df[['PULocationID', 'DOLocationID']]
df = df.mask(df > 263)
df = df.mask(df < 1)
df = df.dropna().sample(n=10000)

###
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
ax = plt.subplot(1, 2, 1)
ax.set_title("Boroughs in NYC")
draw_region_map(ax, sf, shp_dic)
ax = plt.subplot(1, 2, 2)
ax.set_title("Zones in NYC")
draw_zone_map(ax, sf, shp_dic)
plt.show()
###

DO_counts = df.DOLocationID.value_counts()
PU_counts = df.PULocationID.value_counts()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
ax = plt.subplot(1, 2, 1)
ax.set_title("Zones with most pickups")
draw_zone_map(ax, sf, shp_dic, heat=PU_counts.to_dict(), text=PU_counts.head(3).index.tolist())
ax = plt.subplot(1, 2, 2)
ax.set_title("Zones with most drop-offs")
draw_zone_map(ax, sf, shp_dic, heat=DO_counts.to_dict(), text=DO_counts.head(3).index.tolist())
plt.show()

### DBSCAN
fhv_2019_06_DO = df.DOLocationID.to_frame()
fhv_2019_06_PU = df.PULocationID.to_frame()
print(fhv_2019_06_DO.shape, fhv_2019_06_PU.shape)
fhv_2019_06_DO = fhv_2019_06_DO.join(zones.set_index('LocationID'), on='DOLocationID')
fhv_2019_06_PU = fhv_2019_06_PU.join(zones.set_index('LocationID'), on='PULocationID')
print(fhv_2019_06_DO.shape, fhv_2019_06_PU.shape)
X_DO = fhv_2019_06_DO.drop(columns=['DOLocationID']).values
X_PU = fhv_2019_06_PU.drop(columns=['PULocationID']).values
# sb.heatmap(X_DO[:, 0], X_DO[:, 1])
# plt.scatter(X_DO[:, 0], X_DO[:, 1])
# plt.show()

std = StandardScaler()
X_DO = std.fit_transform(X_DO)
X_PU = std.fit_transform(X_PU)
db_DO = DBSCAN(eps=0.23, min_samples=200).fit(X_DO)
db_PU = DBSCAN(eps=0.23, min_samples=200).fit(X_PU)
X_DO = std.inverse_transform(X_DO)
X_PU = std.inverse_transform(X_PU)
core_mask_DO = np.zeros_like(db_DO.labels_, dtype=bool)
core_mask_DO[db_DO.core_sample_indices_] = True
core_mask_PU = np.zeros_like(db_PU.labels_, dtype=bool)
core_mask_PU[db_PU.core_sample_indices_] = True
labels_DO = db_DO.labels_
labels_PU = db_PU.labels_

# Number of clusters in labels_DO, ignoring noise if present.
n_clusters_DO = len(set(labels_DO)) - (1 if -1 in labels_DO else 0)
n_noise_DO = list(labels_DO).count(-1)
n_clusters_PU = len(set(labels_PU)) - (1 if -1 in labels_PU else 0)
n_noise_PU = list(labels_PU).count(-1)

print('No. of clusters in DO: %d' % n_clusters_DO)
print('No. of noise points in DO: %d' % n_noise_DO)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X_DO, labels_DO))
print('No. of clusters in PU: %d' % n_clusters_PU)
print('No. of noise points in PU: %d' % n_noise_PU)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X_PU, labels_PU))

DO_cores = fhv_2019_06_DO.DOLocationID[core_mask_DO].value_counts()
PU_cores = fhv_2019_06_PU.PULocationID[core_mask_PU].value_counts()

fhv_2019_06_DO['Label'] = labels_DO
cluster_counts_DO = fhv_2019_06_DO.Label.value_counts().to_frame(name='Label_counts')
DO_clustered = fhv_2019_06_DO[['DOLocationID', 'Label']][core_mask_DO].drop_duplicates()
DO_clustered = DO_clustered.join(cluster_counts_DO, on='Label')
a = DO_clustered.Label_counts.values.reshape(-1, 1)
b = DO_clustered.Label_counts.unique().reshape(1, -1)
DO_clustered['heat'] = np.sum(a > b, axis=1) + 2
DO_clustered = DO_clustered.set_index('DOLocationID').heat

fhv_2019_06_PU['Label'] = labels_PU
cluster_counts_PU = fhv_2019_06_PU.Label.value_counts().to_frame(name='Label_counts')
PU_clustered = fhv_2019_06_PU[['PULocationID', 'Label']][core_mask_PU].drop_duplicates()
PU_clustered = PU_clustered.join(cluster_counts_PU, on='Label')
a = PU_clustered.Label_counts.values.reshape(-1, 1)
b = PU_clustered.Label_counts.unique().reshape(1, -1)
PU_clustered['heat'] = np.sum(a > b, axis=1) + 2
PU_clustered = PU_clustered.set_index('PULocationID').heat

DO_counts = df.DOLocationID.value_counts()
PU_counts = df.PULocationID.value_counts()

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 16))
ax = plt.subplot(2, 2, 1)
ax.set_title("Zones with most pickups")
draw_zone_map(ax, sf, shp_dic, heat=PU_counts.to_dict(), text=PU_counts.head(1).index.tolist())
ax = plt.subplot(2, 2, 2)
ax.set_title("Zones with most drop-offs")
draw_zone_map(ax, sf, shp_dic, heat=DO_counts.to_dict(), text=DO_counts.head(1).index.tolist())

ax = plt.subplot(2, 2, 3)
ax.set_title("DBSCAN most pickups")
# draw_zone_map(ax, sf, shp_dic, heat=PU_cores.to_dict(), text=PU_cores.head(3).index.tolist())
draw_zone_map(ax, sf, shp_dic, heat=PU_clustered.to_dict(), text=[' '])
ax = plt.subplot(2, 2, 4)
ax.set_title("DBSCAN most drop-offs")
# draw_zone_map(ax, sf, shp_dic, heat=DO_cores.to_dict(), text=DO_cores.head(3).index.tolist())
draw_zone_map(ax, sf, shp_dic, heat=DO_clustered.to_dict(), text=[' '])
plt.show()
