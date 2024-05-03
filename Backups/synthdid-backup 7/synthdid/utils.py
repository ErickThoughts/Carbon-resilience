import pandas as pd
import os
import numpy as np
import re
import copy
from datetime import datetime
from datetime import date
import pypinyin
from pypinyin import pinyin
from sklearn.preprocessing import MinMaxScaler, StandardScaler


city_en2cn = { 'Beijing' :'BJ', 'Tianjin':'TJ', 'Hebei':'HE', 'Shanxi':'SX', 'Inner Mongolia':'NM', 'Liaoning':'LN', 'Jilin':'JL', 'Heilongjiang':'HL',
              'Shanghai':'SH', 'Jiangsu':'JS', 'Zhejiang':'ZJ', 'Anhui':'AH', 'Fujian':'FJ', 'Jiangxi':'JX', 'Shandong':'SD', 'Henan':'HA', 'Hubei':'HB',
              'Hunan':'HN', 'Guangdong':'GD', 'Guangxi':'GX', 'Hainan':'HI', 'Chongqing':'CQ', 'Sichuan':'SC', 'Guizhou':'GZ', 'Yunnan':'YN',
              'Tibet':'XZ', 'Shaanxi':'SN', 'Gansu':'GS', 'Qinghai':'QH', 'Ningxia':'NX', 'Xinjiang':'XJ', 'Wulumuqi':'WLQ', 'Wuhan':'WH',
              'Changchun':'CC', 'Shenzhen':'SZ', 'Guangzhou':'GZ', 'Xian':'XA', 'Quanzhou':'QZ', 'Suzhou':'SUZ', 'Tangshan':'TS', 'Shijiazhuang':'SJZ',
              'Guiyang':'GY', 'Anyang':'AY', 'Haerbin':'HRB', 'Langfang':'LF', 'Nanchang':'NC', 'Shangrao':'SR', 'Guangan':'GAN', 'Suuzhou':'SUU', 
              'Beihai':'BHI', 'Qingdao':'QD' }


provinces = ['北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江', '上海', '江苏',
             '浙江', '安徽', '福建', '江西', '山东', '河南', '湖北', '湖南', '广东', '广西',
             '海南', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏',
             '新疆', '台湾', '香港', '澳门']


state2centre = {  'Beijing':'Beijing', 'Tianjin':'Tianjin', 'Hebei':'Shijiazhuang', 'Shanxi':'Taiyuan', 'Inner Mongolia' :'Huhehaote', 'Liaoning':'Shenyang', 'Jilin':'Changchun',
                            'Heilongjiang':'Haerbin', 'Shanghai':'Shanghai', 'Jiangsu':'Nanjing', 'Zhejiang':'Hangzhou', 'Anhui':'Hefei', 'Fujian':'Fuzhou', 'Jiangxi':'Nanchang',
                            'Shandong':'Qingdao', 'Henan':'Zhengzhou', 'Hubei':'Wuhan', 'Hunan':'Changsha', 'Guangdong':'Guangzhou', 'Guangxi':'Nanning', 'Hainan':'Haikou',
                            'Chongqing':'Chongqing', 'Sichuan':'Chengdu', 'Guizhou':'Guiyang', 'Yunnan' :'Kunming', 'Xizang':'Lasa', 'Shaanxi':'Xian', 'Gansu':'Lanzhou', 'Qinghai':'Xining',
                            'Ningxia':'Yinchuan', 'Xinjiang':'Wulumuqi' }


relative_testing_seq = [ 'Suzhou','Tianjin','Shanghai','Shenzhen','Guangzhou','Beijing','Jilin','Xian','Gansu','Shijiazhuang','Chongqing','Chengdu','Huhehaote','Heilongjiang','Xinjiang','Hainan','Qinghai','Guiyang','Qingdao']
relative_testing_seq2 = [ 'Suzhou','Tianjin', 'Tianjin 1', 'Tianjin 2', 'Shanghai', 'Shenzhen', 'Shenzhen 1', 'Guangzhou', 'Guangzhou 1', 'Beijing',  'Beijing 1', 'Jilin', 'Xian','Gansu','Shijiazhuang','Chongqing','Chengdu','Huhehaote', 'Huhehaote 1', 'Haerbin', 'Heilongjiang','Xinjiang','Hainan','Qinghai','Guiyang','Qingdao']


''' the dictionary indicates on what basis the feature is distributed between cities and provinces '''
FEATURE_RATIO_DIC = {'power':'Power', 
                     'industry':'Indus',
                     'residential' : 'Resi',
                     'trans_people' : 'Trans-p',
                     'trans_cargo' : 'Trans-c',                         
                     'city_economy':'Economy',
                     'temperature':'None', 
                     'history_co2':'Economy'
                     }

    
    
def check_cols(lst_1, lst_2):
    '''
        check any difference between [state names of real-time emission] and [states of feature data]
    '''
    diff_1 = set(lst_1) - set(lst_2)
    diff_2 = set(lst_2) - set(lst_1)
    union_diff = list(diff_1.union(diff_2))
    
    if len(union_diff)>0:
        print(union_diff)
    return union_diff


def divide_over_frs(inputs, feature_meta, raw_len):
    divide_lst = [ inputs[ -raw_len : ] ]
    current_len = 0
    
    for val in feature_meta:
        fr_len = val[0] * val[1]
        update_len = current_len + fr_len
        divide_lst.append( inputs[current_len : update_len] )
        current_len = update_len
    return divide_lst


def log_func(x1, y1, x2, y2):
    A = (y2 - y1) / np.log(x2 / x1)
    B = y1 - A * np.log(x1)
    return lambda x: A * np.log(x) + B


def linear_func(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return lambda x: slope * x + intercept


def sep_lst(list_, n):
    return [list_[i:i+n] for i in range(0, len(list_), n)]


def col_transfer(origin_df):
    try:
        cols = list(map(lambda x : transform_pys(x.split(':')[1]), list(origin_df.columns)))
    except:
        cols = list(map(lambda x : transform_pys(x), list(origin_df.columns)))
    origin_df.columns = cols
    return origin_df
    

def transform_pys(cn_str):
    cn_str = re.sub(r'\s', '',cn_str.strip())
    if '新疆' in cn_str:
        cn_str='新疆'
    if '宁夏' in cn_str:
        cn_str='宁夏'
    if '广西' in cn_str:
        cn_str='广西'
    if '内蒙' in cn_str:
        res_letter = 'Inner Mongolia'
        return res_letter
    if '陕西' in cn_str:
        res_letter='Shaanxi'
        return res_letter
        
    pinyin_list = [py[0] for py in pinyin(cn_str, style=pypinyin.NORMAL)]
    if pinyin_list[-1]=='sheng' or pinyin_list[-1]=='shi':
        pinyin_list=pinyin_list[:-1]
    res_letter = ''.join(pinyin_list).capitalize()
        
    return res_letter


def extract_states(str_):
    pattern = '|'.join(provinces)
    match = re.search(pattern, str_)
    if match:
        province_name = match.group()
        return province_name
    else:
        raise()


def expand_meta_feature(feature_meta):
    feature_meta_new = []
    for val in feature_meta:
        for i in range(val[0]):
            feature_meta_new.append((1,val[1]))
            
    fr_len = len(feature_meta_new) + 1
    return feature_meta_new, fr_len


'''scale can affect the results significantly, take care with how to scale the data'''
def scale_(df_, mode="all"):
    scaler = None
    if mode=='all':
        df_ = np.log(df_ + 10**-5)
    elif mode=='column':
        cities = df_.columns
        for city in cities:
            city_features = np.array(df_[city].values)
            df_[city] = np.log(city_features+10**-8)
    elif mode=='mm':
        scaler = MinMaxScaler()
        df_ = pd.DataFrame(scaler.fit_transform(df_), columns=df_.columns, index=df_.index)
    else:
        pass
    return df_, scaler


def min_max_scale(arr, min_val=0.5, max_val=2):
    arr_min, arr_max = np.min(arr), np.max(arr)
    scaled_arr = ((arr - arr_min) / (arr_max - arr_min)) * (max_val - min_val) + min_val
    return scaled_arr


def standard_scale(arr):
    scaler = StandardScaler()
    scaler.fit(arr)
    arr_scaled = scaler.transform(arr)    
    return arr_scaled


def log_scale(arr, constant):
    arr_scaled = np.log(arr) + constant
    return arr_scaled


def drop_contradic_states(df_, contrad_dic, study_city, tuple_=False, drop_id=0): # id==0 means drop cities, id==1 means drop states
    contra_states = contrad_dic[study_city]
    if tuple_==True:
        contra_states = list(np.unique([c[drop_id] for c in contra_states]))

    if len(contra_states)>0:
        df_.drop(columns=contra_states, inplace=True)
        print('{} dropped states or cities: {}'.format( len(contra_states), ', '.join(contra_states)))
    return df_


def trans_distribute(df_, sector, study_state=None, study_city=None, lock_down_info=None, ratio_rec=None, FEATURE_RATIO_DIC=None, early_return=False):
    input_sector = copy.deepcopy(sector)
    if not study_state==None:
        if not FEATURE_RATIO_DIC==None:
            sector = FEATURE_RATIO_DIC[sector]
        
        sector = sector.title()
        ratio_str = str(list(lock_down_info[lock_down_info['city']==study_city][sector].values)[0])
        ratios = [ra for ra in ratio_str.split(',') if not ra=='']
        ratios =  [float(r_.split('-')[0])/float(r_.split('-')[1]) for r_ in ratios]
        
        ratio_mean = np.mean(ratios)
        if study_city==None:
            study_city = lock_down_info[lock_down_info['state']==study_state]['city'].values[0]
        
        if early_return==True:
            return (study_city, ratio_mean)
        
        if ratio_mean<1:
            df_[study_city] = df_[study_state] * ratio_mean # this will add a column to the original frame
            # df_[study_state] = df_[study_state] * (1-ratio_mean) # change the state value so that the city values are excluded
            
            if not ratio_rec==None:
                if study_state in list(ratio_rec.keys()):
                    ratio_rec.update( { study_state : ratio_rec[study_state] + [ratio_mean] } )
                else:
                    ratio_rec.update( { study_state : [ratio_mean] } )   
            # print('\nsector: {}, convert between {} and {}, with the raio={}, shape:{}'.format(sector, study_state, study_city, round(ratio_mean, 3), df_.shape))
        else:
            # print('\nsector: {}, city and state are the same: {}, shape:{}'.format(sector, study_city, df_.shape))
            df_[study_city] = df_[study_state]
            
    return df_, study_city, ratio_rec


def date_compare(dt1, dt2, mode='larger'):
    if mode=='larger':
        if dt1 >= dt2:
            return dt1
        else:
            return dt2
    if mode=='smaller':
        if dt1 <= dt2:
            return dt1
        else:
            return dt2  
    

def parse_city_string(study_city, lock_down_info):
    city_ = study_city.split('--')[0]
    begin_ = study_city.split('--')[1]
    end_ = study_city.split('--')[2]
    
    state_ = list(lock_down_info[lock_down_info['city']==city_]['state'].values)[0]
    
    return city_, begin_, end_, state_


def gen_city2state(lock_down_info):
    city2state = dict()
    state2city = dict()
    
    cities = list(lock_down_info['city'].values)
    states = list(lock_down_info['state'].values)
    
    for i, c in enumerate(cities):
        city2state.update( {c:states[i]} )
    
    for j, s in enumerate(states):
        if s in list(state2city.keys()):
            state2city.update( { s : state2city[s] + [cities[j]] } )
        else:
            state2city.update( { s : [cities[j]] } )
        
    return city2state, state2city




