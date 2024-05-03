import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from plot import *
from eval import *
from utils import *
from sample_data import read_emission_data, generate_feature_df, check_cols
from variance import fit_time_weights, fit_unit_weights, join_weights, synthetic_diff_in_diff, estimate_se
from results_cal import cal_resilience




def prepare_duration(df_, dates_dic, study_city_key, study_city, begin_, end_, lag_dic, detail, earliest_date='01/01/1997'):
    lockdown_begin = datetime.strptime(begin_, '%Y-%m-%d')
    lockdown_end = datetime.strptime(end_, '%Y-%m-%d')
    earliest_ = datetime.strptime(earliest_date, '%d/%m/%Y')
    print('lock-down begins: {}, lock-down ends: {}.'.format(lockdown_begin, lockdown_end))
    
    city_da_lst = []
    for key_, val_ in dates_dic.items():
        if study_city in key_:
            val_ = [ da_.to_pydatetime() for da_ in val_[0][:-1] ]
            city_da_lst.append(val_)
        else:
            continue
    
    idx = -1
    for da_ in city_da_lst:
        da_token = str(da_[0].strftime('%Y-%m-%d')) + '--' + str(da_[1].strftime('%Y-%m-%d'))
        if da_token in study_city_key:
            idx = city_da_lst.index(da_)
    
    before_ = lag_dic[study_city][0]
    after_ = lag_dic[study_city][1]
    
    if detail==True:
        if idx < len(city_da_lst)-1:
            city_next = city_da_lst[idx+1][0]
            after_ = date_compare(after_, city_next, mode='smaller')
        if idx > 0:
            city_last = city_da_lst[idx-1][1]
            before_ = date_compare(before_, city_last, mode='larger')
        
    df_ = df_[ df_.index >= before_ ]
    df_ = df_[ df_.index <= after_ ]
    PRE_TERM = [earliest_, lockdown_begin]
    POST_TERM = [lockdown_begin + relativedelta(days=1), after_]
    
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
    

def eva_contradics(lock_down_info, sector, split_char='至', threshold=0): # new covid-19 info
    def parse_dates(s, year='1800'):
        start_date_str, end_date_str = s.split(split_char)
        start_date_str = year + '-' + start_date_str.replace('月', '-').replace('日', '')
        end_date_str = year + '-' + end_date_str.replace('月', '-').replace('日', '')
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        return pd.Series((start_date, end_date))

    def ranges_intersect(start1, end1, start2, end2, threshold):
        if start1 <= end2 and end1 >= start2:  # The ranges intersect
            intersection_start = max(start1, start2)
            intersection_end = min(end1, end2)
            intersection_length = intersection_end - intersection_start
            return intersection_length > timedelta(days=7)
        else:
            return False  # The ranges do not intersect
    
    
    origin_cols = list(lock_down_info.columns)[1:]
    lock_down_info[['start_date', 'end_date']] = lock_down_info['date'].apply(parse_dates)
    lock_down_info['start_date'] = lock_down_info.apply(lambda row: row['start_date'].replace(year=row['year']), axis=1)
    lock_down_info['end_date'] = lock_down_info.apply(lambda row: row['end_date'].replace(year=row['year']), axis=1)
    
    lock_down_info = lock_down_info.drop(columns='date')
    lock_down_info = lock_down_info[['start_date', 'end_date'] + origin_cols]
    
    contrad_dic = dict()
    dates_dic = dict()
    lag_dic = dict()
    
    for i, row_i in lock_down_info.iterrows():
        start_i = row_i['start_date']
        end_i = row_i['end_date']
        city_ = row_i['city']
        last_ = end_i - start_i
        city_key = city_ + '--' + start_i.strftime("%Y-%m-%d") + '--' + end_i.strftime("%Y-%m-%d")
        
        lag = row_i['pre_lag']
        pos = row_i['pos_lag']
        
        duration_ = (end_i - start_i).days
        if (sector=='industry' and city_ in ['Hainan', 'Guangxi', 'Guizhou']) or (sector=='power' and city_ in ['Guangxi', 'Qinghai']):
            lag_i = start_i - relativedelta(days=int(lag))
            pos_i = end_i + relativedelta(days=int(pos))         
        else:
            lag_i = start_i - relativedelta(days=int(duration_*1.5))
            pos_i = end_i + relativedelta(days=int(duration_/2)) 
        
        lag_dic.update({city_ : (lag_i, pos_i)})
        
        if city_key in list(dates_dic.keys()):
            dates_ = dates_dic[city_key] + [ (start_i, end_i, last_) ]
            dates_dic.update( {city_key : dates_} )
        else:
            dates_dic.update( {city_key : [(start_i, end_i, last_)]} )

        no_inter_all = True
        for j, row_j in lock_down_info.iterrows():
            if i==j:
                continue
            else:
                start_j = row_j['start_date']
                end_j = row_j['end_date']
                inter_ = ranges_intersect(lag_i, pos_i, start_j, end_j, threshold)
                    
                if inter_==True and not row_j['city']==city_:
                    no_inter_all = False
                    inter_city = row_j['city']
                    inter_state = row_j['state']
                    # print('found intersection: {} --> {}: {}, {}'.format(row_i['city'], row_j['city'], start_j.strftime("%Y-%m-%d"), end_j.strftime("%Y-%m-%d")))
                    
                    if city_key in list(contrad_dic.keys()):
                        contras = contrad_dic[city_key] + [(inter_city, inter_state)]
                        contrad_dic.update( { city_key : contras } )
                    else:
                        contrad_dic.update( { city_key: [(inter_city, inter_state)] } )
        print()
        if no_inter_all==True:
            contrad_dic.update( { city_key: [] } )
                        
    return contrad_dic, lock_down_info, dates_dic, lag_dic

    
    
    
'''=================================================================MAIN CODE==========================================================='''
if __name__ == "__main__":
    outcome_name = 'co2emission'
    sector = 'power' # industry power
    
    FEATURE_PERIODS = [2000, 2020]
    INVERSE = True # False
    ADD_FEATURES = [   ] # 'power', 'industry', 'history_co2', 'temperature', 'city_economy' 
    
    parent_dir = os.path.dirname(os.getcwd())
    info_file = 'data/covid19_info_' + sector + '.csv'
    lock_down_info = pd.read_csv(os.path.join(parent_dir, info_file), encoding='gbk')
    DETAIL_DURATION = False
    
    contrad_dic, lock_down_info, dates_dic, lag_dic = eva_contradics(lock_down_info, sector)
    city2state, state2city = gen_city2state(lock_down_info)
    
    
    ''' process additional features into ideal format '''
    feature_meta = []
    for fr_name in ADD_FEATURES:
        fr_relative_path = 'data/feature data/' + (fr_name + '_feature') + '.csv' 
        feature_meta = generate_feature_df(parent_dir, 
                                               fr_relative_path, 
                                               fr_name, 
                                               feature_meta, 
                                               interpolate=True, 
                                               periods=FEATURE_PERIODS)
    feature_meta, N_VARS = expand_meta_feature(feature_meta)
    print('There are {} features added.... \n'.format(len(feature_meta)))
             
    
    ''' enumerate the cities and do evaluation '''
    syn_res_df = pd.DataFrame( columns=['city', 'syn_coef', 'resi_all_abs', 'resi_in_abs', 'resi_after_abs', 'duration'] )
    
    for i, row in lock_down_info.iterrows():
        study_city = row['city']
        emission_path = os.path.join(parent_dir, ('data/'  + sector + '_emission.csv'))
        emission_resi_ = read_emission_data(emission_path)
        
        if 'people' in sector:
            pass
        else:
            all_cities = list(lock_down_info['city'].values)
            ratio_rec = dict()
            for c in all_cities:
                state_ = city2state[c]
                emission_resi_, _, ratio_rec = trans_distribute(emission_resi_, sector, state_, c, lock_down_info, ratio_rec, FEATURE_RATIO_DIC)
            
            for state_ in list(emission_resi_.columns):
                if state_ in list(ratio_rec.keys()):
                    remain_ = 1 - np.sum(ratio_rec[state_])
                    emission_resi_[state_] = emission_resi_[state_] * remain_
        
        if row['used']==0:
            print('current city {} used for syn-city generation but not for investigation, i.e., not computing their casuals'.format(study_city))
            continue
        
        origin_df = copy.deepcopy(emission_resi_)
        
        '''  drop states/cities that are locked down in the same period ''' 
        study_city_key = list(dates_dic.keys())[i]
        study_city, begin_, end_, study_state = parse_city_string(study_city_key, lock_down_info)
        print('current study city {}, and the state {}'.format(study_city, study_state))
        
        
        if 'people' in sector:
            drops = []
            for c in contrad_dic[study_city_key]:
                if not state2centre[c[1]]==study_city:
                    drops.append( state2centre[c[1]] )
            contrad_dic_ = { study_city : drops }
            drop_contradic_states(emission_resi_, contrad_dic_, study_city)
        else:
            drop_contradic_states(emission_resi_, contrad_dic, study_city_key, tuple_=True, drop_id=0)
            
        
        ''' slice original data based on periods '''
        emission_resi_, PRE_TERM, POST_TERM, BEFORE, lock_begin, lock_end = prepare_duration(emission_resi_, 
                                                                                             dates_dic,
                                                                                             study_city_key,
                                                                                             study_city, 
                                                                                             begin_, 
                                                                                             end_,
                                                                                             lag_dic,
                                                                                             DETAIL_DURATION
                                                                                             ) 
        
        
        emission_resi_, N_NUM, raw_len = deal_add_features(ADD_FEATURES, emission_resi_, study_state, study_city, contrad_dic, FEATURE_RATIO_DIC)
            
            
        ''' scale data '''
        if 'cargo' in sector:
            SCALE = 'mm'
        else:
            SCALE = 'all'
        emission_resi_, scaler = scale_(emission_resi_, SCALE)


        ''' calculate sc_did_values, and extract and export some weights '''
        emission_resi_, control_ys, unit_weights, time_weights, unit_intercept, did_data, syn_res = eval_synthetic_did(emission_resi_,
                                                                    PRE_TERM = PRE_TERM,
                                                                    POST_TERM = POST_TERM,
                                                                    BEFORE = BEFORE,
                                                                    N_NUM = N_NUM,
                                                                    FR_META = feature_meta,
                                                                    N_VARS = N_VARS,
                                                                    RAW_LEN = raw_len,
                                                                    TREATMENT = [study_city],
                                                                    outcome_name = outcome_name,
                                                                    parent_dir = parent_dir
                                                                    )
        
        
        '''inverse scaling'''
        counter_facts = control_ys + unit_intercept
        if INVERSE==True:
            if SCALE=='mm':
                study_idx = list(emission_resi_.columns).index(study_city)
                emission_resi_ = pd.DataFrame(scaler.inverse_transform(emission_resi_), columns=emission_resi_.columns, index=emission_resi_.index)
                min_val, max_val = scaler.data_min_[study_idx], scaler.data_max_[study_idx]
                counter_facts = counter_facts * (max_val-min_val) + min_val
            else:
                emission_resi_ = np.exp(emission_resi_) - 10**-5
                counter_facts = np.exp(counter_facts) - 10**-5
               
        
        '''calculate important figures'''
        resilience, redundancy, att_res, year_last = cal_resilience(emission_resi_, origin_df, counter_facts, study_city, begin_, end_)
        
        ''' export results '''
        syn_res['city'] = study_city
        syn_res['att'] = np.round(att_res[0], 3)
        syn_res['att_in'] = np.round(att_res[1], 3)
        syn_res['att_pos'] = np.round(att_res[2], 3)
        syn_res['att_rela'] = np.round(att_res[3], 3)
        syn_res['att_in_rela'] = np.round(att_res[4], 3)
        syn_res['att_pos_rela'] = np.round(att_res[5], 3)
        syn_res['resi_all_abs'] = np.round(resilience[0], 3)
        syn_res['resi_in_abs'] = np.round(resilience[1], 3)
        syn_res['resi_after_abs'] = np.round(resilience[2], 3)
        syn_res['extreme'] = np.round(redundancy, 3)
        syn_res['year_dff'] = year_last[0]
        syn_res['duration'] = (lock_end - lock_begin).days
        del syn_res['std']
        del syn_res['t']
        del syn_res['p>t']
        tys_last_year = year_last[1]
        syn_res_df = pd.concat([syn_res_df, pd.DataFrame(syn_res, index=[0])])
        
        uniw_name = 'results/' + sector + '_weights/' + study_city + '_u_weights.csv'
        timw_name = 'results/' + sector + '_weights/' + study_city + '_t_weights.csv'
        unit_weights.to_csv(os.path.join(parent_dir, uniw_name))
        time_weights.to_csv(os.path.join(parent_dir, timw_name))
        
        true_ys = np.array(emission_resi_[study_city].values)
        city_res_data = pd.DataFrame( {'cf':counter_facts, 'ts':true_ys, 'last_year':tys_last_year} )
        city_res_data.to_csv(os.path.join(parent_dir, ('results/' + sector + '_city_res/' + study_city + '.csv')))
        
        
        ''' plot figures '''
        inverse_plot_sdid(df_ = emission_resi_,
                               counter_facts = counter_facts,
                               last_vals = tys_last_year,
                               treated = study_city,
                               y_name = outcome_name,
                               x_name = 'date',
                               lock_begin = lock_begin,
                               lock_end = lock_end,
                               raw_len = raw_len,
                               inverse = INVERSE,
                               scaler = scaler
                               )

        # plot_time_weights(time_weights, cut_date)
        # plot_origin(emission_resi_, study_city, outcome_name, 'non-parallel trends', 2000)
        
        # cn_map = os.path.join(parent_dir, 'data/geo_data/gadm41_CHN_1.shp')
        # plot_map_weights(cn_map, unit_weights, study_state, contrad_dic)
        
        # plt.show()
        # plt.close()
        
        im_path = os.path.join(parent_dir, 'results/plots/' + sector + '/cfs_emissions/' + study_city + '.png' )
        plt.savefig(im_path, dpi=300)
        
    
    syn_res_df = syn_res_df.set_index('city')
    counts = {}
    new_index = []
    for city in syn_res_df.index:
        if counts.get(city, 0) > 0:
            new_index.append(f"{city} {counts[city]}")
        else:
            new_index.append(city)
        counts[city] = counts.get(city, 0) + 1
 
    rela_file_path = 'data/relative data/' + sector +'_syn_testing.csv'
    relative_df = pd.read_csv(os.path.join(parent_dir, rela_file_path), encoding='gbk')
    relative_testing_seq = list(relative_df['city'].values)

    syn_res_df.index = new_index
    syn_res_df = syn_res_df.reindex(relative_testing_seq).reset_index()
    syn_res_df = syn_res_df.rename(columns={syn_res_df.columns[0]: 'city'})
        
    for syn_col in syn_res_df.columns:
        relative_df[syn_col] = syn_res_df[syn_col]
    relative_df.to_csv(os.path.join(parent_dir, rela_file_path), mode='w', index=False)
    
    
    ''' evalaute variance '''
    # np.random.seed(1)        
    # did_data = did_data.iloc[:, :-3]
    # did_data['treated'] = did_data['treated'].astype(bool)
    # did_data['after_treatment'] = did_data['after_treatment'].astype(bool)
    # effect = synthetic_diff_in_diff(did_data,
    #                         outcome_col=outcome_name,
    #                         date_col="date",
    #                         state_col="state",
    #                         treat_col="treated",
    #                         post_col="after_treatment")
    #
    # se = estimate_se(did_data,
    #          outcome_col=outcome_name,
    #          date_col="date",
    #          state_col="state",
    #          treat_col="treated",
    #          post_col="after_treatment")
    #
    # print(f"Effect: {effect}")
    # print(f"Standard Error: {se}")
    # print(f"90% CI: ({effect-1.65*se}, {effect+1.65*se})")

    print('finished...')









