import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from eval import *
import geopandas as gpd
import seaborn as sns
from dateutil.relativedelta import relativedelta
from utils import *


def modify_state_names(geo_df, state_weights, study_state, contra_states):
    origin_states = list(state_weights.index)
    geo_df['state_weights'] = 0
    
    for i, row in geo_df.iterrows():
        state_ = row['NAME_1']
        # print(state_, '\t', study_state)
        
        if study_state in state_:
            geo_study_state = state_
            geo_df.loc[i, 'state_weights'] = 0
        elif state_ in contra_states:
            geo_df.loc[i, 'state_weights'] = 0
        else:
            if 'Xinjiang' in state_:
                geo_df.loc[i, 'state_weights'] = state_weights.loc['Xinjiang'].values[0]
            elif 'Nei' in state_:
                geo_df.loc[i, 'state_weights'] = state_weights.loc['Inner Mongolia'].values[0]
            elif 'Ningxia' in state_:
                geo_df.loc[i, 'state_weights'] = state_weights.loc['Ningxia'].values[0]
            elif 'Xizang' in state_:
                continue # keep the value to be 0
            elif 'Macau' in state_ or 'Hong' in state_:
                geo_df.loc[i, 'state_weights'] = np.nan
            else:
                geo_df.loc[i, 'state_weights'] = state_weights.loc[state_].values[0]
    geo_df = geo_df.dropna(how='any')
    return geo_df, geo_study_state
    

def plot_origin(origin_df, treated, x_name, y_name, interven_time):    
    plt.figure(figsize=(10,5))
    plt.plot(origin_df.drop(columns=[treated]), color="C1", alpha=0.3)
    
    plt.plot(origin_df.drop(columns=[treated]).mean(axis=1), lw=3, color="C1", ls="dashed", label="Control Avg.")
    
    plt.plot(origin_df[treated], color="C0", label=treated)
    # plt.vlines(x=interven_time, ymin=40, ymax=10000, linestyle=":", lw=2, label="Policy treatment", color="black")
    plt.legend()
    plt.ylabel(y_name)
    plt.title(x_name);
    

def inverse_plot_sdid(df_, counter_facts, last_vals, treated, y_name, x_name, lock_begin, lock_end, raw_len, inverse, scaler):
    # NOTE if we have multiple variables apart from the outcome, we need to plot the outcome only
    if raw_len>0:
        df_ = df_.iloc[ df_.shape[0]-raw_len : ]
    
    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    
    ax.plot(df_.index, counter_facts, label="Synthetic Control (SDID) + $w_0$", color="C0")
    ax.plot(df_[treated], label=treated, color="C1", alpha=.9)
    ax.plot(df_.index, last_vals, label=treated, color="C2", alpha=.4)

    min_ = np.min( [ np.min(counter_facts), np.min(df_[treated]), np.min(last_vals) ] )
    max_ = np.max( [ np.max(counter_facts), np.max(df_[treated]), np.max(last_vals) ] )
        
    ax.vlines(x=lock_begin, ymin=min_, ymax=max_, linestyle=":", lw=2, label="lock-down", color="black")
    ax.vlines(x=lock_end, ymin=min_, ymax=max_, linestyle=":", lw=2, label="lift", color="black")
    
    ax.legend()
    ax.set_ylabel(y_name);
    

    
def plot_map_weights(cn_map, state_weights_, study_state, contrad_dic, map_color='YlGnBu', state_color='red', check=True): 
    geo_df = gpd.read_file(cn_map)[['NAME_1', 'ENGTYPE_1', 'geometry']]
    contra_states = contrad_dic[study_state]
    
    state_weights_.set_index(state_weights_.columns[0], inplace=True) # state_weights_ = state_weights_.to_frame()
    geo_df, geo_study_state = modify_state_names(geo_df, state_weights_, study_state, contra_states)
    
    if check==True:
        geo4check = geo_df[['NAME_1', 'state_weights']]
        geo4check.to_csv(os.path.join(os.getcwd(), 'geo_res.csv'))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    geo_df.plot(column='state_weights', cmap=map_color, linewidth=0.3, ax=ax, edgecolor='grey')
    
    geo_df.loc[geo_df['NAME_1'] == geo_study_state].plot(color=state_color, ax=ax, legend=True)
    for cs in contra_states:
        geo_df.loc[geo_df['NAME_1'] == cs].plot(color='grey', ax=ax, legend=True)
            
    # add annotations of weights on the map
    for idx, row in geo_df.iterrows():
        if row['NAME_1']==study_state:
            continue
        sc_weight = '{:.2e}'.format(row['state_weights'])
        plt.annotate(text=sc_weight, xy=row['geometry'].centroid.coords[0], horizontalalignment='center')
    ax.axis('off')
    plt.show()    
    

def plot_time_weights(time_weights, cut_date=None):
    if not cut_date==None:
        time_weights = time_weights[time_weights.index >= cut_date]
    x = time_weights.index
    y = time_weights
    
    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    ax.bar(x, y)
    
    ax.set_ylim(0, 1.2*np.max(time_weights.values))
    ax.set_ylabel("Time Weights");
    plt.show()
    

def plot_elast_bars(rnd_pred, model_pred_cols, y_name, t_name, fig_size=(10, 4)):
    fig, axs = plt.subplots(1, len(model_pred_cols), sharey=True, figsize=fig_size) #share the y-axis limits between multiple subplots in a single figure
    for pred_col, ax in zip(model_pred_cols, axs):
        df_bands = elast_by_bands_continuous(rnd_pred, pred_col, y_name, t_name)
        df_bands.plot.bar(ax=ax)        


def plot_causal_curves(rnd_pred, model_pred_cols, y_name, t_name, gain=False, fig_size=(10,6), curve_min_periods=100, steps=100):
    if not isinstance(model_pred_cols, list):
        model_pred_cols = [model_pred_cols]
    
    plt.figure(figsize=fig_size)
    for pred_col in model_pred_cols:
        cumu_elast = cumulative_elasts(rnd_pred, pred_col, y_name, t_name, gain, curve_min_periods, steps)
        x = np.array(range(len(cumu_elast)))
        plt.plot(x/x.max(), cumu_elast, label=pred_col)
    
    plt.hlines(elast_continuous(rnd_pred, y_name, t_name), 0, 1, linestyles="--", color="black", label="Avg. Elast.")
    plt.xlabel("% of Top Elast. Days")
    plt.ylabel("Cumulative Elasticity")
    plt.title("Cumulative Elasticity Curve")
    plt.legend()
    plt.show()
    

def corr_plot(arr_x, arr_y, var_name, im_path, val_pair):
    arr_df = pd.DataFrame( { 'arr_x':arr_x, 'arr_y':arr_y } )
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.set(style="darkgrid")
    sns.lmplot(x="arr_x", y="arr_y", data=arr_df, robust=True, ci=95, scatter_kws={"s": 60, "alpha": 0.3}, line_kws={"color": "red"})
    
    plt.title(var_name + ' correlation results')
    loc_x = np.median(arr_x)
    loc_y = np.median(arr_y)
    plt.text(loc_x, loc_y, f"corr: {np.round(val_pair[0], 2)}, P-value: {np.round(val_pair[1], 3)}", 
             horizontalalignment='center', 
             verticalalignment='center',
             fontsize=15)
    
    plt.savefig(im_path, dpi=300)
    # plt.show()


def heatmap_plot(contingency_table, x_label, y_label, im_path, val_pair):
    plt.figure(figsize=(10, 7))
    sns.heatmap(contingency_table, annot=True, cmap="YlGnBu")    
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.text(0.5, 0.9, f"corr: {np.round(val_pair[0], 2)}, P-value: {np.round(val_pair[1], 3)}", 
             horizontalalignment='center', 
             verticalalignment='center',
             fontsize=15)
    
    if not im_path==None:
        plt.savefig(im_path, dpi=300)
        # plt.show()
    



    
    
    
    
    
    
    
    
    
    
