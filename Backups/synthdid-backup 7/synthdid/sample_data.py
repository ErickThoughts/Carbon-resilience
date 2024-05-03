import pandas as pd
import os
from datetime import datetime
from datetime import date
import pypinyin
from pypinyin import pinyin
import re
from utils import *


def read_emission_data(data_path):
    df_ = pd.read_csv(data_path)
    try:
        df_['date'] = df_['date'].apply(lambda x:datetime.strptime(x, '%d/%m/%Y'))
    except:
        df_['date'] = df_['date'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))
    
    df_core = pd.pivot(df_, index='date', columns='state')
    df_core.columns = df_core.columns.droplevel(0)
    df_core.columns.name = None
    return df_core
    

def process_temp(temp_dir, out_path, periods=None):
    state_dfs = pd.DataFrame()
    provinces_dic = {}
    
    for f_name in os.listdir(temp_dir):
        file = os.path.join(temp_dir, f_name)
        if os.path.isfile(file):
            city_df = pd.read_csv(file, encoding='gbk')
            province = city_df['省份'].values[0]
            city = city_df['城市'].values[0]
            
            city_df = city_df[ ['年份', '省份', '平均气温'] ]
            city_df.columns = ['年份', '省份', city ]
                        
            if not province in list(provinces_dic.keys()):
                provinces_dic.update({ province :  city_df })
            else:
                provinces_dic[province] = pd.concat( [provinces_dic[province], city_df[city] ], axis=1 )
    
    temperature_df = pd.DataFrame()
    for state, df_ in provinces_dic.items():
        if state in [ '香港特别行政区', '台湾省', '西藏自治区' ]:
            continue
        print('current state: {}, there are {} records (i.e., cities) of temperature of {} years'.format(state, len(df_.columns)-2, df_.shape[0]) )
        cities = list(df_.columns[2:])
        df_.columns = ['date', 'state'] + cities
        
        #df_['date'] = df_['date'].apply(lambda x:datetime.strptime(('01/01/'+str(x)), '%d/%m/%Y'))
        df_.set_index(['date', 'state'], inplace=True)
        df_['avg_temperature'] = df_.mean(axis=1)
        df_.drop(columns=cities, inplace=True)
        temperature_df = pd.concat([temperature_df, df_])
    
    temperature_df.reset_index(inplace=True)
    temperature_df.set_index('date', inplace=True)
    temperature_df['state'] = temperature_df['state'].apply(lambda x : transform_pys(x))
    
    if not periods==None:
        temperature_df = temperature_df[ temperature_df.index >= periods[0] ]
        temperature_df = temperature_df[ temperature_df.index <= periods[1] ]
    temperature_df.to_csv(out_path)


# NOTE: this can handle both carbon emission and economy data
def process_his_emission(his_emission_path, out_path, interpolate=False, periods=None):
    his_emission_df = pd.read_csv(his_emission_path, encoding='gbk')
    his_emission_df['state'] = his_emission_df['state'].apply(lambda x : transform_pys(x))
    
    df_subset = his_emission_df[['date', 'state']]
    df_subset['date'] = df_subset['date'].astype(int)
    
    # get the minimum and maximum years for each city
    state_year_range = df_subset.groupby('state')['date'].agg(['min', 'max']).reset_index()
    
    # create a new data frame with all year-city combinations
    df_lst = pd.DataFrame(columns=['date', 'state'])
    for i, row in state_year_range.iterrows():
        state = row['state']
        year_range = range(row['min'], row['max'] + 1)
        year_state_df = pd.DataFrame({'date': year_range, 'state': [state] * len(year_range)})
        df_lst = df_lst.append(year_state_df, ignore_index=True)
    
    # merge with original data frame to get other columns
    feature_df = pd.merge(df_lst, his_emission_df, on=['date', 'state'], how='left')
    if interpolate==True:
        feature_df.interpolate(method='linear', limit_direction='both', inplace=True)
    feature_df.set_index('date', inplace=True)

    if not periods==None:
        feature_df = feature_df[ feature_df.index >= periods[0] ]
        feature_df = feature_df[ feature_df.index <= periods[1] ]
    feature_df.to_csv(out_path)
        

def generate_feature_df(parent_dir, feature_path, fr_name, feature_meta, interpolate=False, periods=None):
    feature_path = os.path.join(parent_dir, feature_path)
    feature_df = pd.read_csv(feature_path)
    feature_num = len(feature_df.columns) - 2
    # print('current processing {}'.format(feature_path))
    
    if interpolate==True:
        feature_df.interpolate(method='linear', limit_direction='both', inplace=True)
    
    feature_df = pd.pivot(feature_df, index='date', columns='state')
    
    #loop over the first-level columns (i.e., the features)
    res_df = pd.DataFrame()
    for f_col in feature_df.columns.levels[0]:
        sub_feature_df = feature_df.loc[ :, f_col]
        res_df = pd.concat([res_df, sub_feature_df])
    
    try:
        res_df = res_df.drop(columns=['Tibet']) # currently drop the column
        print('\tWarning: in {}, found Tibet and remove it'.format(feature_path))
    except:
        # print('status OK: current states do not include Tibet')
        pass
    
    if not periods==None:
        res_df = res_df[ res_df.index >= periods[0] ]
        res_df = res_df[ res_df.index <= periods[1] ]
    
    feature_out_path = os.path.join(os.getcwd(), (fr_name + '_features.csv'))
    res_df.to_csv(feature_out_path)
    
    feature_span = int(res_df.shape[0]/feature_num)
    feature_meta.append((feature_num, feature_span))
    print('\tcurrently additional feature group {}, there are {} features\n'.format(fr_name, feature_num))
    return feature_meta


def seg_monitor_data(raw_path, parent_dir):
    emission_data_path = os.path.join(parent_dir, raw_path)
    emission_df = pd.read_csv(emission_data_path)
    emission_df = emission_df[ emission_df['state']!='Tibet'] # remove Tiebt due to data lackage

    emission_av_df = trans_distribute(emission_df, 'Aviation')
    emission_pwoer_df = trans_distribute(emission_df, 'Power')
    emission_indus_df = trans_distribute(emission_df, 'Industry') 
    
    emission_av_df.to_csv(os.path.join(parent_dir, 'data\\trans_avi_emission.csv'), index=False)
    emission_indus_df.to_csv(os.path.join(parent_dir, 'data\\industry_emission.csv'), index=False)
    emission_pwoer_df.to_csv(os.path.join(parent_dir, 'data\\power_emission.csv'), index=False)
    

def handle_wind_data(origin_df, standard_cols, ratio_dic=None, interpolate=True):
    origin_df['date'] = origin_df['date'].apply(lambda x : datetime.strptime(x, '%d/%m/%Y'))
    origin_df.set_index('date', inplace=True)

    try:
        cols = list(map(lambda x : transform_pys(x.split(':')[1]), list(origin_df.columns)))
    except:
        cols = list(map(lambda x : transform_pys(x), list(origin_df.columns)))
    origin_df.columns = cols
    origin_df = origin_df.reindex(columns=standard_cols)
    origin_df.dropna(axis='columns', how='all', inplace=True)
    
    df_exp = pd.DataFrame()
    for col in list(origin_df.columns):
        state_df = origin_df[[col]] # resample and expand the results to each day
        if interpolate==True:
            df_resampled = state_df.resample('D').interpolate(method='linear')
        else:
            df_resampled = state_df
        
        if not ratio_dic==None:  # distribute data in the years
            ratio_ = ratio_dic[col]
            df_resampled.loc[df_resampled.index<'2020-01-01', col] *= ratio_
            df_resampled.loc[df_resampled.index<'2021-01-01', col] *= ratio_
            df_resampled.loc[df_resampled.index>='2021-01-01', col] *= ratio_
        df_exp = pd.concat([df_exp, df_resampled], axis=1)
        
    df_exp = pd.melt(
         df_exp.reset_index().rename(columns={'index': 'date'}),
         id_vars='date',
         value_name='emission',
         var_name='state')
    
    # swap the position of the data-frame
    swap_order = [df_exp.columns[1], df_exp.columns[0]] + list(df_exp.columns[2:])
    df_exp = df_exp[swap_order]
    return df_exp
    
    

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    lock_down_info = pd.read_csv(os.path.join(parent_dir, 'data/covid19_info.csv'), encoding='gbk')
    standard_cols = list(map(lambda x : transform_pys(x), provinces))
    
    '''pre-process REAL-TIME emission data during lock-down'''
    raw_path = 'data\\raw data\\【原始数据-实时碳排19-22】cn_provincial_emission.csv'
    seg_monitor_data(raw_path, parent_dir)
    
    
    '''pre-process history temperature data'''
    temp_dir_path = os.path.join(parent_dir, 'data\\raw data\\【原始数据】temperature_by_cities')
    temp_out_path = os.path.join(parent_dir, 'data\\feature data\\temperature_feature.csv')
    # process_temp(temp_dir_path, temp_out_path, periods=[2001, 2022])
    
    
    '''pre-process history co2 data'''
    his_emission_path = os.path.join(parent_dir, 'data\\raw data\\city emissions.csv')
    his_out_path = os.path.join(parent_dir, 'data\\feature data\\history_co2_feature.csv')
    # process_his_emission(his_emission_path, his_out_path, interpolate=True, periods=[2001, 2020])
    
    
    '''pre-process city economy data'''
    city_economy_path = os.path.join(parent_dir, 'data\\raw data\\city economics.csv')
    eco_out_path = os.path.join(parent_dir, 'data\\feature data\\city_economy_feature.csv')
    # process_his_emission(city_economy_path, eco_out_path, interpolate=True, periods=[2001, 2020])
    
    
    '''pre-process residential and transportation data from WIND'''
    res_ratio = pd.read_csv(os.path.join(parent_dir, 'data\\raw data\\city sector ratios.csv'), encoding='gbk')
    res_ratio['state'] = res_ratio['state'].apply(lambda x:transform_pys(x))
    ratio_dic = res_ratio.set_index(res_ratio.columns[0])[res_ratio.columns[1]].to_dict()
    
    
    ''' ----> residential co2 data (electricity consumption)'''
    # resi_srcs = [ '社会用电量'  ]
    # resi_df = pd.DataFrame()
    # for src in resi_srcs:
    #     emission_path = os.path.join(parent_dir, 'data\\raw data\\【原始数据】' + src + '.csv')
    #     current_resi_df = pd.read_csv(emission_path, encoding='utf-8')
    #     current_resi_df = handle_wind_data(current_resi_df, ratio_dic, standard_cols, distribute=True)
    #     if resi_df.empty:
    #         resi_df = current_resi_df
    #     else:
    #         resi_df['emission'] += current_resi_df['emission']
    # resi_df.to_csv(os.path.join(parent_dir, 'data\\residential_emission.csv'), index=False)
    
    
    ''' ----> transportation people co2 data combine central city data and state data '''
    trans_people = pd.DataFrame()
    # emission_path = os.path.join(parent_dir, 'data/raw data/【原始数据】公路客运量.csv')
    # trans_people = pd.read_csv(emission_path, encoding='utf-8')
    
    emission_path = os.path.join(parent_dir, 'data/raw data/【原始数据】中心城市客运量.csv')
    trans_people = pd.read_csv(emission_path, encoding='utf-8')

    current_states = list(map(lambda x : transform_pys(x), list(trans_people.columns)))
    # trans_people = handle_wind_data(trans_people, current_states[1:])
    # trans_people.to_csv(os.path.join(parent_dir, 'data\\trans_people_emission.csv'), index=False)
    
    
    ''' ----> transportation cargo co2 data'''
    trans_cargo = pd.DataFrame()
    emission_path = os.path.join(parent_dir, 'data\\raw data\\【原始数据】公路货运量.csv')
    
    trans_cargo = pd.read_csv(emission_path, encoding='utf-8')

    # trans_cargo = handle_wind_data(trans_cargo, standard_cols)
    # trans_cargo.to_csv(os.path.join(parent_dir, 'data\\trans_cargo_emission.csv'), index=False)
    
    
    
    
    
    
    

