import pandas as pd
import os
import numpy as np
from utils import min_max_scale, standard_scale, scale_, log_scale
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import spearmanr
from scipy.stats import pearsonr



def eval_resi_vals(tys_period, cfs_period, cap_=150):
    baseline = np.min( (np.min(tys_period), np.min(cfs_period)) )
    
    cfs_baseline = cfs_period - baseline
    tys_baseline = tys_period - baseline
    
    # calculate resilience
    upper_curve = np.maximum(cfs_baseline, tys_baseline)
    lower_curve = np.minimum(cfs_baseline, tys_baseline)
    
    area_lower = np.trapz(lower_curve)
    area_upper = np.trapz(upper_curve)
    resilience = area_lower / area_upper
    
    # calculate redundancy
    diff = tys_baseline - cfs_baseline
    max_dif = np.max(np.abs(diff))
    max_dif_loc = np.argmax(np.abs(diff))
    
    tys_diff_val = tys_baseline[max_dif_loc]
    cfs_diff_val = cfs_baseline[max_dif_loc]
    redundancy = 1 - max_dif/(max(tys_diff_val, cfs_diff_val))
    
    return resilience, redundancy
    

def cal_resilience(df_, origin_df, counter_facts, treated, begin_, end_):
    position_begin = df_.index.get_loc(begin_)
    position_end = df_.index.get_loc(end_)

    df_period = df_[ df_.index >= begin_ ]    
    tys_period = df_period[treated].values
    cfs_period = counter_facts[position_begin:]
    
    last_ = str(df_.index[0])
    last_year = str(int(last_.split('-')[0]) - 1)
    last_year_date = '-'.join([last_year] + last_.split('-')[1:])
    
    df_last = origin_df[ origin_df.index >= last_year_date ]
    df_last = df_last.iloc[ : len(df_) ]
    tys_last_year = df_last[treated].values
    
    df_period_in = df_[ df_.index >= begin_ ]
    df_period_in = df_period_in[ df_period_in.index <= end_ ]
    tys_period_in = df_period_in[treated].values
    cfs_period_in = counter_facts[ position_begin : position_end+1 ]
    
    df_period_after = df_[ df_.index >= end_  ]
    tys_period_after = df_period_after[treated].values
    cfs_period_after = counter_facts[ position_end : ]
    
    # note here we currently ignore the area and only collect the resilience of the all, pre and after periods
    resi_all_abs, redun_all = eval_resi_vals(tys_period, cfs_period)
    resi_in_abs, _ = eval_resi_vals(tys_period_in, cfs_period_in)
    resi_af_abs, _ = eval_resi_vals(tys_period_after, cfs_period_after)
    
    att_ = np.mean(tys_period) - np.mean(cfs_period) # true emissions - counterfactual emissions
    att_in = np.mean(tys_period_in) - np.mean(cfs_period_in)
    att_pos = np.mean(tys_period_after) - np.mean(cfs_period_after)
    
    att_rela = att_/np.mean(tys_period)
    att_rela_in = att_in/np.mean(tys_period_in)
    att_rela_pos = att_pos/np.mean(tys_period_after)
    
    year_diff = np.mean(tys_period) - np.mean(tys_last_year)
    
    return (resi_all_abs, resi_in_abs, resi_af_abs), redun_all, (att_, att_in, att_pos, att_rela, att_rela_in, att_rela_pos), (year_diff, tys_last_year)



'''================================================================Draw resilience plot==========================================================='''
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    sector = 'power' # power industry
    
    resi_file_path = 'data/relative data/' + sector + '_syn_testing.csv'
    resi_df = pd.read_csv(os.path.join(parent_dir, resi_file_path))
    
    resi_indicators = ['resi_all_abs', 'resi_in_abs', 'resi_after_abs', 'extreme'] # 'time_conv', 'resource'
    other_features = ['city', 'type', 'att', 'att_in', 'att_pos']
    
    resi_df = resi_df[other_features + resi_indicators]
    resi_df = resi_df.dropna(axis=1, how='any')
    
    # pre-process and merge duplicate rows
    df4regions = pd.DataFrame(columns=['city'] + resi_indicators)
    skip_rows = 0
    
    for i, _ in resi_df.iterrows():
        i = i + skip_rows
        if i > len(resi_df)-1:
            break
        
        i_row = resi_df.iloc[i] 
        city_i = i_row['city']
        i_row_vals = i_row[resi_indicators].values
        
        sim_count = 1 # reset sim_count
        for j, j_row in resi_df.iterrows():
            if i==j:
                continue
            else:
                j_row_vals = j_row[resi_indicators].values
                city_j = j_row['city']
                if city_i in city_j:
                    sim_count += 1
                    skip_rows += 1
                    i_row_vals = i_row_vals + j_row_vals
        
        i_row_vals = i_row_vals/sim_count
        new_i_row = pd.DataFrame([[city_i] + list(i_row_vals)], columns=df4regions.columns)
        df4regions = pd.concat([df4regions, new_i_row], ignore_index=True)
    
    resi_df.to_csv(os.path.join(parent_dir, 'results/' + sector + '_resilience.csv'))
    df4regions.to_csv(os.path.join(parent_dir, 'results/' + sector + '_radar.csv'))


    # plot radar figures
    for i, row in df4regions.iterrows():
        im_path = os.path.join(parent_dir, 'results/plots/' + sector + '/radars/' + row['city'] + '.png')
    
        row_vals = row[resi_indicators]
        current_radars = list(row_vals.values)
        angles = np.linspace(0,2*np.pi, len(current_radars), endpoint=False)
    
        current_radars = np.concatenate((current_radars, [current_radars[0]])) 
        angles = np.concatenate((angles, [angles[0]]))
        labels = np.concatenate((resi_indicators, [resi_indicators[0]]))
    
        fig = plt.figure(facecolor='white')
        ax = plt.subplot(111, polar=True)
    
        ax.set_ylim(0, 1)
        ax.plot(angles, current_radars, color='g', linewidth=2)
        ax.fill(angles, current_radars, facecolor='g', alpha=0.25)
        ax.set_thetagrids(angles*180/np.pi, labels)
        ax.grid(True)       
    
        plt.savefig(im_path, dpi=300)
        # plt.show()
        plt.close()
    
    
    # draw initial scatter plots to show the shape
    num_groups = 4
    for indi_col in resi_indicators:
        im_path = os.path.join(parent_dir, 'results/plots/' + sector + '/quantile' + '_' + indi_col + '.png' )
    
        val_df = pd.concat([ resi_df['city'], resi_df[indi_col] ], axis=1)
        val_df[indi_col+'_group'] = pd.qcut(val_df[indi_col], q=num_groups, duplicates='drop')  # .rank(method='first')
        # val_df[indi_col+'_group'] = pd.cut(val_df[indi_col], bins=num_groups)
    
        val_df_sorted = val_df.sort_values(indi_col)
    
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='city', y=indi_col, hue=indi_col+'_group', data=val_df_sorted, palette='Set2', s=100)
    
        plt.title(indi_col + ' quantile groups')
        plt.xlabel('city')
        plt.ylabel(indi_col)
        plt.xticks(rotation=45)
    
        plt.savefig(im_path, dpi=300)
        # plt.show()
        plt.close()
    

    
    
    
    
    
    
    
    

