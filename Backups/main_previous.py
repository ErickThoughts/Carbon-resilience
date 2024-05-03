import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta

from plot import *
from eval import *
from utils import *

from sample_data import read_emission_data, generate_feature_df, check_cols


def prepare_duration(df_, lock_down_info, study_city, mode, full_release_date, earliest_date='01/01/1997'):
    lockdown_begin = list(lock_down_info[lock_down_info['city']==study_city]['begin'].values)[0]
    lockdown_begin = datetime.strptime(lockdown_begin, '%d/%m/%Y')
    
    lockdown_end = list(lock_down_info[lock_down_info['city']==study_city]['end'].values)[0]
    lockdown_end = datetime.strptime(lockdown_end, '%d/%m/%Y')
    print('lockdown begins: {}, lockdown ends: {}.'.format(lockdown_begin, lockdown_end))
    
    full_release_date = datetime.strptime(full_release_date, '%d/%m/%Y')
    earliest_ = datetime.strptime(earliest_date, '%d/%m/%Y')
    
    # analyze the reduction of carbon emission due to lock-down, 3 months data before lock-down to the end of lock-down
    if mode=='lock_down' :
        after_ = lockdown_end + relativedelta(months=1)
        before_ = lockdown_begin - relativedelta(months=2) # use the 10 month temporal data to evaluate time weights
        df_ = df_[ df_.index >= before_ ]
        df_ = df_[ df_.index <= after_ ]
        PRE_TERM = [earliest_, lockdown_begin]
        POST_TERM = [lockdown_begin + relativedelta(days=1), after_]
    else:
        df_ = df_[ df_.index > full_release_date]
        PRE_TERM = [0, lockdown_end]
        POST_TERM = [full_release_date, -1]
    
    BEFORE = df_.index[0] # here no additional features, 'BEFORE' records when the real-time emission begins for evaluating time weights
    return df_, PRE_TERM, POST_TERM, BEFORE, lockdown_begin, lockdown_end
       
    
def deal_add_features(add_features, df_, study_state, study_city, contrad_dic, FEATURE_RATIO_DIC, take_mean=True):
    row_provinces = list(df_.columns)
    raw_len = df_.shape[0]
    
    # CONCATENATE additional features
    if len(add_features)>0:
        for f, fr_name in enumerate(add_features):
            print('currently process feature {}  {}...'.format(f, fr_name))
            feature_df = pd.read_csv(os.path.join(os.getcwd(), (fr_name + '_features.csv')))
            feature_df['date'] = feature_df['date'].apply(lambda x:datetime.strptime(('01/01/'+str(x)), '%d/%m/%Y'))
            feature_df.set_index('date', inplace=True)
            
            # distribute feature data based on city-province ratio
            if not study_state==study_city:
                feature_df, study_city = trans_distribute(feature_df, fr_name, study_state, lock_down_info, FEATURE_RATIO_DIC)
            
            drop_contradic_states(feature_df, contrad_dic, study_state)
            unmatched_states = check_cols(list(feature_df.columns), row_provinces)
            
            if len(unmatched_states)==0:
                feature_df = feature_df.reindex(columns=row_provinces) # ensure the order of columns
                if take_mean==True:
                    mean_df = pd.DataFrame(columns=feature_df.columns)
                    mean_df.loc[feature_df.index[0]] = feature_df.mean().values
                    feature_df = mean_df
                df_ = pd.concat([feature_df, df_])
    return df_, df_.shape[1], raw_len
    

def eva_contradics(lock_down_info, year='2022', split_char='至'): # new covid-19 info
    def parse_dates(s, year):
        start_date_str, end_date_str = s.split(split_char)
        start_date_str = year + '-' + start_date_str.replace('月', '-').replace('日', '')
        end_date_str = year + '-' + end_date_str.replace('月', '-').replace('日', '')
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        return pd.Series((start_date, end_date))

    lock_down_info[['start_date', 'end_date']] = lock_down_info['date'].apply(parse_dates)
    


'''=================================================================MAIN CODE==========================================================='''
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    lock_down_info = pd.read_csv(os.path.join(parent_dir, 'data\\lock_down_info.csv'))
    
    outcome_name = 'co2emission'
    sector = 'power' # industry trans_people trans_cargo residential power
    
    study_state = 'Qinghai' # Chongqing Shanghai Jilin Xinjiang Hubei Guangdong
    study_city = 'Qinghai' # Wulumuqi  Changchun  Wuhan Shenzhen    
    
    contrad_dic = {'Shanghai':['Jilin'], 'Jilin':['Shanghai'], 'Xinjiang':['Shanghai','Jilin'], 'Qinghai':['Xinjiang'],
                   'Hubei':[], 'Chongqing':['Xinjiang'], 'Guangdong':['Shanghai','Jilin']}
    
    # contrad_dic = {'Shanghai':['Changchun'], 'Changchun':['Shanghai'], 'Wulumuqi':['Shanghai','Changchun'], 
    #                'Hubei':[], 'Chongqing':['Wulumuqi'], 'Guangdong':['Shanghai','Jilin'], 'Shenzhen' :[ 'Shanghai' , 'Changchun' ] }
    
    full_release_date = '05/12/2022'

    FEATURE_PERIODS = [2000, 2020]
    SCALE = 'all' # column, none all
    ADD_FEATURES = [   ] # 'power', 'industry', 'history_co2', 'temperature', 'city_economy' 
    
    
    '''
        process additional features into ideal format
    '''
    feature_meta = []
    for fr_name in ADD_FEATURES:
        fr_relative_path = 'data\\feature data\\' + (fr_name + '_feature') + '.csv' 
        feature_meta = generate_feature_df(parent_dir, 
                                               fr_relative_path, 
                                               fr_name, 
                                               feature_meta, 
                                               interpolate=True, 
                                               periods=FEATURE_PERIODS)
    feature_meta, N_VARS = expand_meta_feature(feature_meta)
    print('There are {} features\n'.format(len(feature_meta)))
    
    
    '''
        LOAD real-time emission data and handle states that went through lock-down in the same period
    '''
    emission_resi_ = read_emission_data(os.path.join(parent_dir, ('data\\'  + sector + '_emission.csv')))
    print(emission_resi_)
    
    '''
        distribute co2 emission of a province to the investigated city
    '''
    if not study_city==study_state:
        emission_resi_, study_city = trans_distribute(emission_resi_, sector, study_state, study_city, lock_down_info, FEATURE_RATIO_DIC)
    
    '''
        drop contradictory states/cities that are locked down in the same period
    '''
    drop_contradic_states(emission_resi_, contrad_dic, study_state)
    
    '''
        slice the original data based on periods
            'PRE_TERM' indicates the earliest record of feature data, e.g., 1997, 2000...
            'BEFORE' is used to evaluate time weights (from the earliest date of real-time emission data)
    '''
    emission_resi_, PRE_TERM, POST_TERM, BEFORE, lock_begin, lock_end = prepare_duration(emission_resi_, lock_down_info, study_city, 'lock_down', full_release_date) 
    
    emission_resi_, N_NUM, raw_len = deal_add_features(ADD_FEATURES, emission_resi_, study_state, study_city, contrad_dic, FEATURE_RATIO_DIC)
    
    # if we drop the province and only remain the city
    # emission_resi_.drop(columns=[study_state], inplace=True)
    
    '''
        scale the data and save the version before and after scaling for cross comparison
    '''
    emission_resi_.to_csv(os.path.join(os.getcwd(), 'df_before_scale.csv'))
    emission_resi_ = scale_(emission_resi_, SCALE)
    emission_resi_.to_csv(os.path.join(os.getcwd(), 'df_all_scale.csv'))
    

    '''
        calculate sc_did_values, and extract and export some weights
    '''
    TREATMENT = [study_city]
    out_unit_weights = pd.DataFrame()
    # out_unit_weights = pd.read_csv(os.path.join(os.getcwd(), 'outer_ws.csv'))
    # out_unit_weights = out_unit_weights.drop(columns=TREATMENT)
    # vals = np.squeeze(out_unit_weights.T.values)
    # out_unit_weights= pd.Series(vals, name="unit_weights", index=out_unit_weights.columns)
    
    control_ys, unit_weights, time_weights, unit_intercept, melt_df, did_data, did_model = eval_synthetic_did(emission_resi_,
                                                                PRE_TERM = PRE_TERM,
                                                                POST_TERM = POST_TERM,
                                                                BEFORE = BEFORE,
                                                                N_NUM = N_NUM,
                                                                FR_META = feature_meta,
                                                                N_VARS = N_VARS,
                                                                RAW_LEN = raw_len,
                                                                TREATMENT = TREATMENT,
                                                                outcome_name = outcome_name,
                                                                parent_dir = parent_dir,
                                                                outer_ws = out_unit_weights)
    
    unit_weights.to_csv(os.path.join(os.getcwd(), 'results\\unit_weights.csv'))
    time_weights.to_csv(os.path.join(os.getcwd(), 'results\\time_weights.csv'))
    
    '''
        plot the results
    '''
    cut_date = lock_begin - relativedelta(months=4)
    cut_date = None
    
    plot_time_weights(time_weights, cut_date)
    
    # plot_origin(emission_resi_, TREATMENT[0], outcome_name, 'non-parallel trends', 2000)
    plot_sdid(emission_resi_, control_ys, unit_intercept, TREATMENT[0], outcome_name, 'date', lock_begin, lock_end, raw_len, cut_date=cut_date)
    
    # cn_map = os.path.join(parent_dir, 'data/geo_data/gadm41_CHN_1.shp')
    # unit_weights = pd.read_csv(os.path.join(os.getcwd(), 'results/unit_weights.csv'))
    # plot_map_weights(cn_map, unit_weights, study_state, contrad_dic)
    plt.show()
    
    print(unit_weights)














