import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from utils import *
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster.tests.test_affinity_propagation import n_clusters
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns




def read_cluster_data(cluster_path, lock_down_info, cluster_trans_dic, years=list(range(2000, 2020)), since=2010, scale=False):
    data = pd.read_csv(cluster_path)
    data.iloc[ : , 1: ] = data.iloc[ : , 1: ].interpolate(method='linear', limit_direction='both')
    if scale:
        data.iloc[ : , 1: ] = scale_(data.iloc[ : , 1: ],  mode='all')
    
    studied_ = []
    for i, row in lock_down_info.iterrows():
        if row['used']==1:
            studied_.append(row['city'])
    
    dfs = {}
    for column in data.columns[ 1: ]: # extract the column as a DataFrame and store it in the dictionary
        temp_df = pd.concat( [ data['state'], pd.DataFrame(data[column]) ], axis=1 ).set_index('state')
        val_name = temp_df.columns[-1]
        
        transformed_df = pd.DataFrame()
        sector = cluster_trans_dic[column]
        for sc in studied_:
            state_ = list(lock_down_info[lock_down_info['city']==sc]['state'].values)[0]
            city_, ratio = trans_distribute(temp_df, sector, state_, sc, lock_down_info, early_return=True)
            assert city_==sc
            
            if not city_==state_:
                city_df = temp_df.loc[state_] * ratio
                city_df.index = [city_] * len(city_df)
                temp_df = pd.concat( [ temp_df, city_df ] )                
                temp_df.loc[state_] = temp_df.loc[state_] * (1-ratio)
                print('\nsector: {}, convert between {} and {}, with the raio={}'.format(sector, state_, city_, ratio))
                
        temp_df.index.name = 'state'
        groups = temp_df.groupby('state')
        
        for city_name, group_df in groups:
            group_df = group_df.reset_index()[ [val_name] ]
            group_df.columns = [city_name]
            transformed_df = pd.concat( [transformed_df, group_df], axis=1 )
            
        transformed_df.insert(loc=0, column='year', value=years)
        transformed_df = transformed_df.set_index('year')
        transformed_df = transformed_df[transformed_df.index>since]
        
        all_city_states = list(transformed_df.columns)
        city_num = len(all_city_states)
        dfs[val_name] = transformed_df

        sep_feature_pth = os.path.join(cluster_dir, 'export features/' + val_name +'.csv' )
        transformed_df.to_csv(sep_feature_pth)
    return dfs, all_city_states, city_num, studied_


def cal_dtw(cluster_df, city_num):
    distances = np.zeros((city_num, city_num))
    for i in range(city_num):
        for j in range(city_num):
            arr1 = cluster_df.iloc[:, i].values.reshape(-1, 1)
            arr2 = cluster_df.iloc[:, j].values.reshape(-1, 1)
            distances[i, j], _ = fastdtw(arr1, arr2, dist=euclidean)
    return distances


def cal_avg_dtw(cluster_dfs, city_num, npy_pth):
    dtws = []
    for feature_name, cluster_df in cluster_dfs.items():
        dtw_mat = cal_dtw(cluster_df, city_num)
        dtws.append(dtw_mat)
    mean_dtw = np.mean(dtws, axis=0)
    np.save(os.path.join( parent_dir, npy_pth), mean_dtw)
    return mean_dtw


def eval_best_n_clusters(mean_dtw, city_num):
    best_n_clusters = 2  # 最少从2个类开始
    best_score = -1  # 轮廓系数范围在-1到1之间
    for n_clusters in range(2, city_num):
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(mean_dtw)

        score = silhouette_score(mean_dtw, labels, metric='precomputed')
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
    return best_n_clusters
    
    
'''=================================================================MAIN CODE==========================================================='''
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    cluster_dir = os.path.join( parent_dir, 'data/cluster data'  )
    cluster_trans_dic = { 'Pop':'pop', 'GDP':'Economy', 'GDP-1':'Economy', 'GDP-2':'Economy', 'GDP-3':'Economy',
                          'elec_produce':'Power', 'water_e':'Power', 'fire_e':'Power', 'oil_consump':'Power',
                          'electri_consump':'Power', 'coal_consump':'Power', 'gas_consump':'Power', 'energy_intensity':'Power'}
    
    # some meta settings
    sector = 'power' # power industry trans_people trans_cargo residential
    scale = False # False True
    mode = 'temp_cluster'  # temp_cluster  explore
    lock_down_info = pd.read_csv(os.path.join(parent_dir, 'data/covid19_info.csv'), encoding='gbk')
    
    if mode=='temp_cluster':
        n_clusters = 5 # 4 -1
        cluster_path = ''
        if sector=='power':
            cluster_path = os.path.join( cluster_dir, 't-cluster power consump.csv' )
        elif sector=='industry':
            cluster_path = os.path.join( cluster_dir, 't-cluster power economy.csv' )
        elif sector=='residential':
            cluster_path = os.path.join( cluster_dir, 't-cluster power economy.csv' )
        elif sector=='trans_people':
            cluster_path = os.path.join( cluster_dir, 't-cluster power economy.csv' )
        elif sector=='trans_cargo':
            cluster_path = os.path.join( cluster_dir, 't-cluster power economy.csv' )
        else:
            pass
        
        cluster_dfs, all_city_states, city_num, studied_ = read_cluster_data(cluster_path, lock_down_info, cluster_trans_dic, scale=scale)
        
        # calculate temporal distance
        npy_pth='data/cluster data/avg_dtw.npy' 
        mean_dtw = cal_avg_dtw(cluster_dfs, city_num, npy_pth)
        
        # load calculated distance for clustering
        mean_dtw = np.load(os.path.join(parent_dir, npy_pth))
        
        if n_clusters==-1:
            n_clusters = eval_best_n_clusters(mean_dtw, city_num)
            print('automatically evaluate {} clusters.'.format( n_clusters ))
        else:
            print('use manually set {} clusters.'.format(n_clusters))
            
        # run k-means
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(mean_dtw)
        score = silhouette_score(mean_dtw, labels, metric='precomputed')
        print( 'k-means finished, the final score is {}'.format(score) )


        cluster_res = pd.DataFrame( { 'city':all_city_states, 'label':labels } )
        cluster_res.to_csv( os.path.join(cluster_dir, 'results_t-cluster.csv') )
    
        # Use PCA to reduce dimensionality of your data to 2 dimensions
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(mean_dtw)
        
        # plot the results
        cmap = plt.get_cmap('Set2', np.max(labels) + 1)
        sizes = np.ones_like(reduced_data[:,0]) * 20
        
        for i, city in enumerate(all_city_states):
            if city in studied_:                
                sizes[i] = 80
                city_abs = city_en2cn[city]
                plt.text(reduced_data[i, 0], reduced_data[i, 1], city_abs)
                print(city, '--', city_abs)

        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap=cmap, s=sizes)
        plt.show()
    
    # explore mode (Not clustering)
    else:
        years_before = 3
        explor_pd = pd.read_csv( os.path.join( cluster_dir, '20-22 features.csv' ) ).dropna(axis=1).set_index(['state'])
        vars_ = list(explor_pd.columns)[1:]
        unique_states = list(np.unique(explor_pd.index))
        
        sequence = np.arange(len(explor_pd)) // years_before
        
        explor_avg_pd = explor_pd.groupby('state').mean().iloc[ : , 1: ]
        explor_avg_pd.to_csv( os.path.join( cluster_dir, 'temp.csv' ) )
        
        # plotting 20-22 results
        n_row = len(vars_)//5
        fig, axes = plt.subplots(n_row, 5, figsize=(int(4*5*n_row), int(5*n_row)))
        colors = sns.color_palette('pastel', 5*n_row)
        
        for i, ax in enumerate(axes.flatten()):
            var = vars_[i]
            var_pd = pd.DataFrame(explor_avg_pd[var], columns=[var]).reset_index()
            var_pd['type'] = var
            df_melt = var_pd.melt(id_vars='type', value_vars=var, var_name='variables', value_name='values')
            
            sns.boxplot(x='variables', y='values', hue='type', data=df_melt, color=colors[i], ax=ax)
            
            for city_ in city4investi:
                dot_value = explor_avg_pd.loc[ city_, var ]
                ax.scatter(x=[var], y=[dot_value], color='red', zorder=2)
                ax.text(var, dot_value, city_en2cn[city_], color='black', ha='right', va='bottom')
            
        plt.show()
        
        
        # for city_ in cityofinterest:
        #     cityofinterest_df = pd.DataFrame([explor_avg_pd.loc[city_]] * len(explor_avg_pd)).reset_index(drop=True)
        #     cityofinterest_df['type'] = city_
        #     explor_avg_pd = pd.concat([explor_avg_pd, cityofinterest_df])
        #
        # explor_avg_pd = explor_avg_pd.reset_index()
        # df_melt = explor_avg_pd.melt(id_vars='type', value_vars=vars_, var_name='variables', value_name='values')
        
        # plt.figure(figsize=(10, 6))
        # sns.boxplot(x='variables', y='values', hue='type', data=df_melt)
        # plt.show()
        #
        # # Add a point for the GDP of City2
        # for city_ in cityofinterest: 
        #     for var in vars_:
        #         plt.scatter(x=[var], y=[explor_avg_pd.loc[ city_, var ]], color='red', zorder=2)
    
    
    
    
    
    
    
    
    












