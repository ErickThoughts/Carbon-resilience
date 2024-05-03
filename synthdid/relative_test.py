import os
import pandas as pd
import numpy as np
from utils import *
from plot import corr_plot, heatmap_plot
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import copy



'''=================================================================MAIN CODE==========================================================='''
# qcut() produce bins with the same number of samples
# cut() produce bins with same ranges, each range (category) has different samples

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    
    sector = 'power' # power industry
    sec_dic = {'industry':'power', 'power':'industry'}
    
    dependent_vars = [ 'resi_all_abs', 'resi_in_abs', 'resi_after_abs', 'extreme' ]
    n_bins = 2 # 0 3 2    X var, if perform Chi-square testing, cut independent variables (x-variables) into 'n_bins'
    start_var = 'duration'
    
    binary = False
    scale = True # False True
    model = 'spearman' # pearson spearman
    
    threshold_contin = 0.02 # 0.05 0.01
    threshold_disc = 0.11

    specific_independents = []
    
    rela_file_path = 'data/relative data/' + sector +'_syn_testing.csv'
    sig_pd_spear = pd.DataFrame(columns=['depen_var', 'factor', 'corr', 'sig'])
    sig_pd_chiqs = pd.DataFrame(columns=['depen_var', 'factor', 'corr', 'sig'])
    
    for dependent_var in dependent_vars:
        print('currently y evaluated: ', dependent_var)
        relative_df = pd.read_csv(os.path.join(parent_dir, rela_file_path))
        relative_df = relative_df.dropna(axis=1, how='all')

        other_resi_file_path = 'data/relative data/' + sec_dic[sector] +'_syn_testing.csv'
        supple_resi_df = pd.read_csv(os.path.join(parent_dir, other_resi_file_path))
        supple_resi_df = supple_resi_df[dependent_vars]
        supple_resi_df.columns = [ (sec_dic[sector] + '_' + col) for col in supple_resi_df.columns ]
        
        relative_df = pd.concat([relative_df, supple_resi_df], axis=1)
        len_all = len(relative_df)

        num_independents = list(relative_df.columns).index(start_var)
        if len(specific_independents)==0:
            feature4tests = list(relative_df.columns[ num_independents : ])
        else:
            feature4tests = specific_independents
        
        relative_df.to_csv(os.path.join(parent_dir, '_' + sector + '_spearman.csv' ))  
        
        # if not binary and n_bins>0:
        #     relative_df = relative_df[relative_df['type']!=2]
        
        if n_bins>0:
            if binary:
                arr_y = np.array(relative_df['type'].values)
                arr_y = np.where(arr_y == 2, 2, 0)
            else:
                arr_y = np.array(relative_df['type'].values)
        else:
            arr_y = np.array(relative_df[dependent_var].values)
            if scale==True:
                arr_y = np.log(arr_y + 10**-6)
            
        # copy array Y for reset default operation 
        arr_y_copy = copy.deepcopy(arr_y)
                
        # processing independent variables
        # feature4tests = ['area', 'oil_produce']
        feature4tests = ['ubr','ifcity']
        all_na_cols = dict()
        for var_col in feature4tests:
            print('\n\tindependent x evaluated:', var_col)
            
            feature_vals = relative_df[var_col]
            if feature_vals.isna().any():
                mask = feature_vals.isna()
                na_ids = np.where(mask)[0]
                arr_y = arr_y[~mask]
                feature_vals = feature_vals.dropna()
                
                all_na_cols.update({var_col:len(na_ids)})
                print('\tthe independent var {} contain {} empty values, filtering x and y...'.format(var_col, len(na_ids)))
                
                # check if there are enough samples
                if (len_all-len(na_ids))<10:
                    print('\tthe independent var {} is skipped due to insufficient samples...'.format(var_col))
                    arr_y = arr_y_copy
                    continue
                
            if n_bins>0:
                arr_x = pd.qcut(feature_vals, q=n_bins, labels=False, duplicates='drop')
            else:
                arr_x = np.array(feature_vals.values)
                if scale==True:
                    arr_x = np.log(arr_x + 10**-6)      
            
            # performing correlation or Chi-square testing
            if n_bins==0: # continuous testing
                im_path = os.path.join(parent_dir, 'results/plots/' + sector + '/spearman/' + dependent_var + '_' + var_col + '.png' )
                if model=='spearman':
                    corr, p_value = spearmanr(arr_y, arr_x)
                else:
                    corr, p_value = pearsonr(arr_y, arr_x)
                
                if p_value < threshold_contin:
                    # corr_plot(arr_x, arr_y, var_col, im_path, (corr, p_value))
                    print(f"\tsignificant variable {var_col}, corr: {corr}, P-value: {p_value}")
                else:
                    print(f"\tnone significant variable {var_col}, corr: {corr}, P-value: {p_value}")
                    
                # record significance and correlation vals
                sig_pd_spear.loc[len(sig_pd_spear)] = [dependent_var, var_col, corr, p_value]
            
            else: # chi-square testing
                im_path = os.path.join(parent_dir, 'results/plots/' + sector + '/chisquare/' + dependent_var + '_' + var_col + '.png' )
                contingency_table = pd.crosstab(arr_x, arr_y)
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
                if p_value <= threshold_disc:
                    heatmap_plot(contingency_table, 'city_type', 'dis_'+var_col, im_path, (chi2, p_value))
                    print(f"\tsignificant variable {var_col}, chi-: {chi2}, P-value: {p_value}")
                else:
                    print(f"\tnone significant variable {var_col}, chi-: {chi2}, P-value: {p_value}")
                
                sig_pd_chiqs.loc[len(sig_pd_chiqs)] = [dependent_var, var_col, chi2, p_value]
            
            # reset Y array for the current dependent var
            plt.close()
            arr_y = arr_y_copy
    
    
    if len(sig_pd_chiqs)>0:
        if binary:
            sig_pth = os.path.join(parent_dir, 'results/' + sector + '_binary_chis.csv' )
        else:
            sig_pth = os.path.join(parent_dir, 'results/' + sector + '_chis11.csv' )
        # sig_pd_chiqs = sig_pd_chiqs[sig_pd_chiqs['sig']<threshold_disc]
        sig_pd_chiqs.to_csv(sig_pth, mode='w', index=False)
    
    elif len(sig_pd_spear)>0:
        sig_pth = os.path.join(parent_dir, 'results/' + sector + '_spearman11.csv' )
        # sig_pd_spear = sig_pd_spear[sig_pd_spear['sig']<threshold_contin]
        sig_pd_spear.to_csv(sig_pth, mode='w', index=False)
    else:
        print('no dataframe contains values, check the codes')
    print('\n evaluation finished...')
    
    
    
    




   
    
    
    
    
    
    
    












