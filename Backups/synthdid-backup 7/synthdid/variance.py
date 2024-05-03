import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import statsmodels.formula.api as smf
import cvxpy as cp
from toolz import curry, partial



def calculate_regularization(data, outcome_col, year_col, state_col, treat_col, post_col):
    
    n_treated_post = data.query(post_col).query(treat_col).shape[0]
    
    first_diff_std = (data
                      .query(f"~{post_col}")
                      .query(f"~{treat_col}")
                      .groupby(state_col)
                      [outcome_col]
                      .diff()
                      .std())
    
    return n_treated_post**(1/4) * first_diff_std


def make_random_placebo(data, state_col, treat_col):

    control = data.query(f"~{treat_col}")
    
    states = control[state_col].unique()
    placebo_state = np.random.choice(states)
    
    return control.assign(**{treat_col: control[state_col] == placebo_state})


def fit_unit_weights(data, outcome_col, date_col, state_col, treat_col, post_col):
    zeta = calculate_regularization(data, outcome_col, date_col, state_col, treat_col, post_col)
    pre_data = data.query(f"~{post_col}")
    
    # pivot the data to the (T_pre, N_co) matrix representation
    y_pre_control = (pre_data
                     .query(f"~{treat_col}")
                     .pivot(date_col, state_col, outcome_col))
    
    # group treated units by time periods to have a (T_pre, 1) vector.
    y_pre_treat_mean = (pre_data
                        .query(f"{treat_col}")
                        .groupby(date_col)
                        [outcome_col]
                        .mean())
    
    # add a (T_pre, 1) column to the begining of the (T_pre, N_co) matrix to serve as intercept
    T_pre = y_pre_control.shape[0]
    X = np.concatenate([np.ones((T_pre, 1)), y_pre_control.values], axis=1) 
    
    # estimate unit weights. Notice the L2 penalty using zeta
    w = cp.Variable(X.shape[1])
    objective = cp.Minimize(cp.sum_squares(X@w - y_pre_treat_mean.values) + T_pre*zeta**2 * cp.sum_squares(w[1:]))
    constraints = [cp.sum(w[1:]) == 1, w[1:] >= 0]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)
    
    return pd.Series(w.value[1:], # remove intercept
                     name="unit_weights",
                     index=y_pre_control.columns)


def fit_time_weights(data, outcome_col, date_col, state_col, treat_col, post_col):
        control = data.query(f"~{treat_col}")
        
        # pivot the data to the (T_pre, N_co) matrix representation
        y_pre = (control
                 .query(f"~{post_col}")
                 .pivot(date_col, state_col, outcome_col))
        
        # group post-treatment time period by units to have a (1, N_co) vector.
        y_post_mean = (control
                       .query(f"{post_col}")
                       .groupby(state_col)
                       [outcome_col]
                       .mean()
                       .values)
        
        # add a (1, N_co) vector of 1 to the top of the matrix, to serve as the intercept.
        X = np.concatenate([np.ones((1, y_pre.shape[1])), y_pre.values], axis=0)
        
        # estimate time weights
        w = cp.Variable(X.shape[0])
        objective = cp.Minimize(cp.sum_squares(w@X - y_post_mean))
        constraints = [cp.sum(w[1:]) == 1, w[1:] >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)

        return pd.Series(w.value[1:], # remove intercept
                         name="time_weights",
                         index=y_pre.index)
        
        
def join_weights(data, unit_w, time_w, date_col, state_col, treat_col, post_col):

    return (
        data
        .set_index([date_col, state_col])
        .join(time_w)
        .join(unit_w)
        .reset_index()
        .fillna({time_w.name: data[post_col].mean(),
                 unit_w.name: data[treat_col].mean()})
        .assign(**{"weights": lambda d: (d[time_w.name]*d[unit_w.name]).round(10)})
        .astype({treat_col:int, post_col:int}))
    
    
def synthetic_diff_in_diff(data, outcome_col, date_col, state_col, treat_col, post_col):
    # find the unit weights
    unit_weights = fit_unit_weights(data,
                                    outcome_col=outcome_col,
                                    date_col=date_col,
                                    state_col=state_col,
                                    treat_col=treat_col,
                                    post_col=post_col)

    # find the time weights
    time_weights = fit_time_weights(data,
                                    outcome_col=outcome_col,
                                    date_col=date_col,
                                    state_col=state_col,
                                    treat_col=treat_col,
                                    post_col=post_col)

    # join weights into DiD Data
    did_data = join_weights(data, unit_weights, time_weights,
                            date_col=date_col,
                            state_col=state_col,
                            treat_col=treat_col,
                            post_col=post_col)
    
    formula = f"{outcome_col} ~ {post_col}*{treat_col}"
    did_model = smf.wls(formula, data=did_data, weights=did_data["weights"] + 4e-5).fit()
    
    return did_model.params[f"{post_col}:{treat_col}"]


def estimate_se(data, outcome_col, date_col, state_col, treat_col, post_col, bootstrap_rounds=100, seed=0, njobs=4):
    np.random.seed(seed=seed)
    
    sdid_fn = partial(synthetic_diff_in_diff,
                      outcome_col=outcome_col,
                      date_col=date_col,
                      state_col=state_col,
                      treat_col=treat_col,
                      post_col=post_col)
    
    effects = Parallel(n_jobs=njobs)(delayed(sdid_fn)(make_random_placebo(data, state_col=state_col, treat_col=treat_col))
                                     for _ in range(bootstrap_rounds))
    
    return np.std(effects, axis=0)



            
            
