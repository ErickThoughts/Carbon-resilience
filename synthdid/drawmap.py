import copy
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import re
from urllib import parse
import hashlib
import requests

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler



def modify_state_names(geo_df):
    for i, row in geo_df.iterrows():
        state_ = row['NAME_1']
        
        if 'Xinjiang' in state_:
            geo_df.loc[i, 'NAME_1'] = 'Xinjiang'
        elif 'Nei' in state_:
            geo_df.loc[i, 'NAME_1'] = 'Inner Mongolia'
        elif 'Ningxia' in state_:
            geo_df.loc[i, 'NAME_1'] = 'Ningxia'
                
    geo_df = geo_df.dropna(how='any')
    return geo_df


def measure_resi(df, resi_cols, measured, threshold=0.2):
    df2process = copy.deepcopy(df)
    remain_lst = [ col for col in resi_cols if col != measured ]
    
    df2process['avg_others'] = df2process[remain_lst].mean(axis=1) 
    df2process[measured + '_diff'] = df2process[measured] - df2process['avg_others']
    df[measured + '_adv'] = df2process[measured + '_diff'] > threshold
    return df


def map_shapes(adv):
    # shapes = row['resi_all_abs_adv', 'resi_in_abs_adv', 'resi_after_abs_adv', 'extreme_adv']
    if adv==True:
        return ('o', 'relative strength')
    else:
        return ('o', 'no relative strength')

def map_sizes(base, indicator):
    return base + 100*indicator


if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    sector = 'industry'
    resi_indis = ['resi_all_abs', 'resi_in_abs', 'resi_after_abs', 'extreme', 'inte_resi' ]
    camps = [ 'Blues', 'Greys', 'BuGn', 'YlGn',  'Pastel2', 'Set3', 'tab20c', 'Accent', 'YlGnBu' ]
    
    cn_map = os.path.join(parent_dir, 'data/CHN_adm/CHN_adm1.shp')
    tw_map = os.path.join(parent_dir, 'data/TWN_adm/TWN_adm1.shp')
    
    resilience_pth = os.path.join(parent_dir, 'results/' + sector + '_geo.csv')
    phe_pth = os.path.join(parent_dir, 'results/' + sector + '_resilience.csv')
    
    
    phe_df = pd.read_csv(phe_pth)
    region_counts = phe_df['state'].value_counts()
    region_counts = region_counts.sort_index()
    region_counts = region_counts.reset_index()
    region_counts.columns = ['state', 'counts']
    count_dic = region_counts.set_index('state')['counts'].to_dict()
    
    
    resi_df = pd.read_csv(resilience_pth, encoding='gbk')
    for indi in resi_indis:
        resi_df = measure_resi(resi_df, resi_indis, indi, threshold=0.15)    
    
    # get dot coordinates
    resi_df['lat'] = resi_df['lat'].apply(lambda x : np.mean( [float(d) for d in x.split('-')]))
    resi_df['lng'] = resi_df['lng'].apply(lambda x : np.mean( [float(d) for d in x.split('-')]))
    
    
    regions = gpd.GeoDataFrame.from_file(cn_map)[['NAME_1', 'ENGTYPE_1', 'geometry']]
    tw_regions = gpd.GeoDataFrame.from_file(tw_map)[['NAME_1', 'ENGTYPE_1', 'geometry']]
    regions = gpd.GeoDataFrame(pd.concat([regions, tw_regions], ignore_index=True))
    
    regions = modify_state_names(regions)
    print(regions.shape)
    
    
    # regions['coords'] = regions['geometry'].apply(lambda x: x.representative_point().coords[0])
    # fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # ax.set_extent([73, 135, 18, 54])  # Set the extent to cover the area of China
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.LAND, edgecolor='black')

    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel("Longitude", fontsize=18)  # Adjust font size if needed
    ax.set_ylabel("Latitude", fontsize=18) 
    
    # regions = pd.merge(regions, counts_, on='NAME_1', how='left')
    regions['counts'] = list(np.ones(regions.shape[0]))
    regions['weights'] = list(np.ones(regions.shape[0]))
    
    for i, row in regions.iterrows():
        region = row['NAME_1']
        if region in list(count_dic.keys()):
            regions.loc[i, 'counts'] = count_dic[region]
        else:
            regions.loc[i, 'counts'] = 0
            print(region)
    
    
    print(regions.shape)
    # scaler = MinMaxScaler()
    # regions['counts'] = scaler.fit_transform(regions['counts'].values.reshape(-1, 1)).flatten()
    
    '''DRAW THE MAP'''
    regions.plot(ax=ax, column='weights', cmap=camps[0], legend=False, linewidth=0.3, edgecolor='k')
                
    
    resi_df.to_csv(os.path.join(parent_dir, 'test.csv'))
    
    all_resi_df = resi_df.iloc[: , [0, 1, 2, 3, 8]]
    all_resi_df.columns = ['region', 'lng', 'lat', 'indicator', 'advantage']
    # OrRd
    
    in_resi_df = resi_df.iloc[: , [0, 1, 2, 4, 9]]
    in_resi_df.columns = ['region', 'lng', 'lat', 'indicator', 'advantage']
    # PuRd
    
    pos_resi_df = resi_df.iloc[: , [0, 1, 2, 5, 10]]
    pos_resi_df.columns = ['region', 'lng', 'lat', 'indicator', 'advantage']
    # YlGnBu
    
    ex_resi_df = resi_df.iloc[: , [0, 1, 2, 6, 11]]
    ex_resi_df.columns = ['region', 'lng', 'lat', 'indicator', 'advantage']
    # BuGn
    
    inte_resi_df = resi_df.iloc[: , [0, 1, 2, 7, 12]]
    inte_resi_df.columns = ['region', 'lng', 'lat', 'indicator', 'advantage']
    # BuPu
        
    '''NOTE THE SELECTED DATA FRAME'''
    selected_df = ex_resi_df
    dot_colmap = plt.cm.BuGn
    mark_size = 50
    
    
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    for idx, row in selected_df.iterrows():
        ax.scatter(row['lng'], row['lat'], 
                   marker = map_shapes(row['advantage'])[0], 
                   c = [dot_colmap(norm(row['indicator']))],  # Wrap the color in a list
                   s = map_sizes(mark_size, row['indicator']))
    
    sm = plt.cm.ScalarMappable(cmap=dot_colmap, norm=norm)
    sm.set_array([])
    
    # Add the colorbar to the bottom right
    cax = fig.add_axes([0.82, 0.16, 0.03, 0.2])  # Adjust the values for exact positioning and size
    plt.colorbar(sm, cax=cax, orientation='vertical')
    
    plt.show()
    fig_pth = os.path.join(parent_dir, '_' + sector + '_geo.png')
    fig.savefig(fig_pth, dpi=600, bbox_inches='tight')
    plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    