import pandas as pd
import numpy as np
from toolz import curry
import os
from sdid_model import SynthDID
import statsmodels.formula.api as smf
from utils import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler



def eval_sdid_att(did_data, dependent_var, independent_vars, link_char='*'):
    def regression_formula(y_var, x_vars):
        formula = y_var + ' ~ '
        formula += link_char.join(x_vars)
        return formula
    
    formula_ = regression_formula(dependent_var, independent_vars)
    has_nas = did_data.isna().any().any()
    did_model = smf.wls(formula_, data=did_data, weights=did_data['weights'] + 1e-10).fit()
    tables = did_model.summary().tables[1]
    
    table_res = { 'city':None, 'syn_coef':tables.data[-1][1], 'std':tables.data[-1][2], 't':tables.data[-1][3], 'p>t':tables.data[-1][4] }
    
    print(did_model.summary().tables[1])
    return did_model, table_res


def eval_synthetic_did(data, zeta_type='base', single=True, **kwargs):
    PRE_TERM = kwargs['PRE_TERM']
    POST_TERM = kwargs['POST_TERM']
    BEFORE = kwargs['BEFORE']
    N_NUM = kwargs['N_NUM']
    N_VARS = kwargs['N_VARS']
    feature_meta = kwargs['FR_META']
    RAW_LEN = kwargs['RAW_LEN']
    TREATMENT = kwargs['TREATMENT']
    outcome_name = kwargs['outcome_name']
    parent_dir = kwargs['parent_dir']
    
    # generate the synthetic control did model and evaluate the weights
    sdid = SynthDID(data, PRE_TERM, POST_TERM, BEFORE, feature_meta, RAW_LEN, TREATMENT)
    sdid.fit(zeta_type=zeta_type)
    
    unit_weights = sdid.hat_omega
    time_weights = sdid.hat_lambda
    unit_intercept = sdid.intercept_omega
        
    if single==True:
        data = data[ data.shape[0]-RAW_LEN : ]
        data = data[ data.index >= BEFORE ] # NOTE the data includes both pre and post period
        N_VARS = 1
        
    # transform the original data-frame
    melt_df = pd.melt(
         data.reset_index().rename(columns={'index': 'date'}),
         id_vars='date',
         value_name=outcome_name,
         var_name='state')
    
    melt_df['treated'] = melt_df['state']==TREATMENT[0] # find and label the city that is treated
    melt_df['after_treatment'] = melt_df['date'].between(POST_TERM[0], POST_TERM[-1]) # find and label periods after the treatment
    
    # join the evaluated weights to the melt_df
    did_data = join_weights(melt_df, unit_weights, time_weights,
                            year_col='date',
                            region_col='state',
                            treat_col='treated',
                            post_col='after_treatment',
                            N_NUN=N_NUM, 
                            N_VARS=N_VARS,
                            checkpath=os.getcwd())
    
    # evaluate the ATE using the weights
    dependent_var = outcome_name
    independent_vars = ['after_treatment', 'treated']
    did_model, syn_res = eval_sdid_att(did_data, dependent_var, independent_vars)
    
    weights_ = list(did_data['weights'].values +1e-10)
    weights_ = sep_lst(weights_, (POST_TERM[-1]-PRE_TERM[0]).days + 1)
    
    control_ys = list(data.drop(columns=TREATMENT[0]).values @ unit_weights) # why not the overall weights?
    
    return data, control_ys, unit_weights, time_weights, unit_intercept, did_data, syn_res
 
        
def join_weights(did_df, unit_w, time_w, year_col, region_col, treat_col, post_col, N_NUN, N_VARS, checkpath=None):
    time_w_name = time_w.name
    unit_w_name = unit_w.name
    
    time_w = pd.DataFrame(time_w).reset_index().rename(columns={'index': year_col}).set_index(year_col)
    unit_w = pd.DataFrame(unit_w).reset_index().rename(columns={'index': region_col}).set_index(region_col)
    
    did_df = did_df.set_index([year_col, region_col])
    transformed_df = pd.DataFrame()

    sub_dfs = np.array_split(did_df, N_NUN)
    time_ws = np.array_split(time_w, N_VARS)
    
    for j in range(N_VARS): # for each variable, we iterate all regions, extract the corresponding var-matrix
        temp_transform_df = pd.DataFrame()
        for sub_i, subject_df in enumerate(sub_dfs): # NOTE every sub-df corresponds to a state, each iteration generates a dataframe of a feature over all states
            var_dfs = np.array_split(subject_df, N_VARS) # extract the var-matrix of eacth state and concatenate them into a data-frame
            current_var_df = var_dfs[j]
            temp_transform_df = pd.concat([temp_transform_df, current_var_df])
        # each temp_transform_df refers to a feature over all states
        temp_transform_df = temp_transform_df.join(time_ws[j])
        temp_transform_df = temp_transform_df.join(unit_w).reset_index()
        temp_transform_df = temp_transform_df.fillna({time_w_name: temp_transform_df[post_col].mean(), unit_w_name: temp_transform_df[treat_col].mean()})
        temp_transform_df[unit_w_name] = np.abs(temp_transform_df[unit_w_name].values)
        temp_transform_df[time_w_name] = np.abs(temp_transform_df[time_w_name].values)
        
        transformed_df = pd.concat([transformed_df, temp_transform_df])
        
    transformed_df = transformed_df.assign(**{'weights': lambda d : (d[time_w_name] * d[unit_w_name]).round(10)})
    transformed_df= transformed_df.astype({treat_col:int, post_col:int})
    if not checkpath==None:
        transformed_df.to_csv(os.path.join(checkpath, 'transformed_df.csv'))
    return transformed_df


def predict_elast(model, df_, treatment_name, h=0.01): # note this is especially applied to regression model and its data frame
    df_elast = df_.assign(**{treatment_name : df_[treatment_name] + h})
    elast_col = (model.predict(df_elast) - model.predict(df_)) / h
    return elast_col

# curry transforms a function with multiple arguments into a series of functions that each take a single argument.
@curry
def elast_continuous(data, y, t):# line coeficient for the one variable linear regression   
    reg_num = np.sum((data[t] - data[t].mean())*(data[y] - data[y].mean()))
    reg_den = np.sum((data[t] - data[t].mean())**2)
    return reg_num / reg_den
        

def elast_by_bands_continuous(df, pred_col, outcome, treatment, bands=10): #makes quantile partitions
    df_bands = df.assign(**{f'{pred_col}_band':pd.qcut(df[pred_col], q=bands)})
    df_bands = df_bands.groupby(f'{pred_col}_band')
    df_bands = df_bands.apply(elast_continuous(y=outcome, t=treatment)) # by default axis=0 i.e., by columns
    return df_bands               
   

def cumulative_elasts(df, pred_col, y, t, gain=False, min_periods=30, steps=100):
    size = df.shape[0]
    ordered_df = df.sort_values(pred_col, ascending=False).reset_index(drop=True) # orders the df by the `pred_col` column
    
    # create a sequence of row numbers that will define our Ks, the last item is all rows (the size of the dataset)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    
    # cumulative computes elasticity. First for the top min_periods units. then for the top (min_periods + step*1), then (min_periods + step*2) and so on
    cecs = []
    for rows in n_rows:
        current_elast = elast_continuous(ordered_df.head(rows), y, t)
        if gain==True:
            cecs.append(current_elast * (rows/size))
        else:
            cecs.append(current_elast)
    cecs = np.array(cecs)
    return cecs


    
    
    
