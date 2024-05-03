import pandas as pd
import numpy as np
from datetime import datetime
from optimizer import Optimize
from utils import *


class SynthDID(Optimize):
    """
    Synthetic Difference in Differences
    df              : pandas.DataFrame
    pre_term        : term before treatment
    post_term       : term after treatmetn
    treatment_unit  : treatment columns names list

    [example]
    df = fetch_CaliforniaSmoking()
    sdid = SynthDID(df, [1970, 1979], [1980, 1988], ["California"])
    sdid.fit()
    """

    def __init__(self, df, pre_term, post_term, before, feature_meta, raw_len, treatment_unit:list, random_seed=0, **kwargs):
        # first val
        self.df = df
        self.pre_term = pre_term
        self.post_term = post_term
        self.before_cut = before
        
        self.raw_len = raw_len
        self.feature_meta = feature_meta
        self.random_seed = random_seed
        
        # able to change
        self.treatment = treatment_unit
        self.control = [col for col in df.columns if col not in self.treatment]
        self._divide_data()

        # params
        self.hat_zeta = None
        self.base_zeta = None
        self.hat_omega_ADH = None
        self.hat_omega = None
        self.intercept_omega = None
        self.hat_lambda = None
        self.intercept_lambda = None
        
        self.hat_omega_ElasticNet = None
        self.hat_omega_Lasso = None
        self.hat_omega_Ridge = None
        self.sdid_se = None
        self.sc_se = None
        self.did_se = None


    def _divide_data(self):
        ids = self.df.index
        
        if(len(self.pre_term)==1):
            self.Y_pre_c = self.df.loc[self.pre_term[0], self.control]
            self.Y_pre_t = self.df.loc[self.pre_term[0], self.treatment]
        else:
            pre_slice = []
            pre_slice = [True if bool_id>=self.pre_term[0] and bool_id<=self.pre_term[1] else False for bool_id in ids]
            self.Y_pre_c = self.df.loc[pre_slice, self.control]
            self.Y_pre_t = self.df.loc[pre_slice, self.treatment]
            
            if not self.before_cut==None:
                self.df = self.df.iloc[ (self.df.shape[0] - self.raw_len) : ] # get the only real-time emission data
                ids = self.df.index
                
                pre_slice = [True if bool_id>=self.before_cut and bool_id<=self.pre_term[1] else False for bool_id in ids]
                self.Y_pre_c_beforecut = self.df.loc[pre_slice, self.control]
             
        if(len(self.post_term)==1):
            self.Y_post_c = self.df.loc[self.post_term[0], self.control]
            self.Y_post_t = self.df.loc[self.post_term[0], self.treatment]
        else:
            post_slice = [True if bool_id>=self.post_term[0] and bool_id<=self.post_term[1] else False for bool_id in ids]
            self.Y_post_c = self.df.loc[post_slice, self.control]
            self.Y_post_t = self.df.loc[post_slice, self.treatment]            

        self.n_treat = len(self.treatment)
        self.n_post_term = len(self.Y_post_t)


    def fit(
        self,
        model="all",
        zeta_type="base",
        force_zeta=None,
        sparce_estimation=False,
        cv=5,
        cv_split_type="KFold",
        candidate_zata=[],
        n_candidate=20,
        sc_v_model="linear",
        additional_X=pd.DataFrame(),
        additional_y=pd.DataFrame(),
        ):

        # estimate zeta
        self.base_zeta = self.est_zeta(self.Y_pre_c, self.feature_meta, (self.raw_len-self.n_post_term) )
        if zeta_type == "base":
            self.zeta = self.base_zeta
        elif zeta_type == "grid_search":
            self.zeta = self.grid_search_zeta(
                cv=cv,
                n_candidate=n_candidate,
                candidate_zata=candidate_zata,
                split_type=cv_split_type,
            )[0]
        elif zeta_type == "bayesian_opt":
            self.zeta = self.bayes_opt_zeta(cv=cv, split_type=cv_split_type)[0]
        else:
            print(f"your choice :{zeta_type} is not supported.")
            self.zeta = self.base_zeta

        if force_zeta != None:
            self.zeta = force_zeta
        
        # estimate omega (unit weights) and lambda (time weights) 
        self.hat_omega, self.intercept_omega = self.est_omega(self.Y_pre_c, self.Y_pre_t, self.zeta)
        
        # self.hat_lambda, self.intercept_lambda = self.est_lambda(self.Y_pre_c, self.Y_post_c)
        self.hat_lambda, self.intercept_lambda = self.est_lambda(self.Y_pre_c_beforecut, self.Y_post_c)
        
        self.hat_omega_ADH = self.est_omega_ADH(
            self.Y_pre_c,
            self.Y_pre_t,
            additional_X=additional_X,
            additional_y=additional_y)
        
        if sparce_estimation:
            self.hat_omega_ElasticNet = self.est_omega_ElasticNet(self.Y_pre_c, self.Y_pre_t)
            self.hat_omega_Lasso = self.est_omega_Lasso(self.Y_pre_c, self.Y_pre_t)
            self.hat_omega_Ridge = self.est_omega_Ridge(self.Y_pre_c, self.Y_pre_t)


    def did_potentical_outcome(self):
        """
        return potential outcome
        """
        Y_pre_c = self.Y_pre_c.copy()
        Y_pre_t = self.Y_pre_t.copy()
        Y_post_c = self.Y_post_c.copy()
        Y_post_t = self.Y_post_t.copy()
    
        if type(Y_pre_t) != pd.DataFrame:
            Y_pre_t = pd.DataFrame(Y_pre_t)
    
        if type(Y_post_t) != pd.DataFrame:
            Y_post_t = pd.DataFrame(Y_post_t)
    
        Y_pre_t["did"] = Y_pre_t.mean(axis=1)
    
        base_trend = Y_post_c.mean(axis=1)
    
        Y_post_t["did"] = base_trend + (
            Y_pre_t.mean(axis=1).mean() - Y_pre_c.mean(axis=1).mean()
        )
    
        return pd.concat([Y_pre_t["did"], Y_post_t["did"]], axis=0)
    
    
    def sc_potentical_outcome(self):
        return pd.concat([self.Y_pre_c, self.Y_post_c]).dot(self.hat_omega_ADH)
    
    
    # def sparceReg_potentical_outcome(self, model="ElasticNet"):
    #     Y_pre_c_intercept = self.Y_pre_c.copy()
    #     Y_post_c_intercept = self.Y_post_c.copy()
    #     Y_pre_c_intercept["intercept"] = 1
    #     Y_post_c_intercept["intercept"] = 1
    #
    #     if model == "ElasticNet":
    #         s_omega = self.hat_omega_ElasticNet
    #     elif model == "Lasso":
    #         s_omega = self.hat_omega_Lasso
    #     elif model == "Ridge":
    #         s_omega = self.hat_omega_Ridge
    #     else:
    #         print(f"model={model} is not supported")
    #         return None
    #     return pd.concat([Y_pre_c_intercept, Y_post_c_intercept]).dot(s_omega)
    #
    #
    # def sparce_sdid_potentical_outcome(self, model="ElasticNet"):
    #     Y_pre_c_intercept = self.Y_pre_c.copy()
    #     Y_post_c_intercept = self.Y_post_c.copy()
    #     Y_pre_c_intercept["intercept"] = 1
    #     Y_post_c_intercept["intercept"] = 1
    #
    #     if model == "ElasticNet":
    #         s_omega = self.hat_omega_ElasticNet
    #     elif model == "Lasso":
    #         s_omega = self.hat_omega_Lasso
    #     elif model == "Ridge":
    #         s_omega = self.hat_omega_Ridge
    #     else:
    #         print(f"model={model} is not supported")
    #         return None
    #
    #     base_sc = Y_post_c_intercept @ s_omega
    #     pre_treat_base = (self.Y_pre_t.T @ self.hat_lambda).values[0]
    #     pre_control_base = Y_pre_c_intercept @ s_omega @ self.hat_lambda
    #
    #     post_outcome = base_sc + pre_treat_base - pre_control_base
    #
    #     return pd.concat([Y_pre_c_intercept.dot(s_omega), post_outcome], axis=0)


    # def estimated_params(self, model="sdid"):
    #     Y_pre_c_intercept = self.Y_pre_c.copy()
    #     Y_post_c_intercept = self.Y_post_c.copy()
    #     Y_pre_c_intercept["intercept"] = 1
    #     Y_post_c_intercept["intercept"] = 1
    #     if model == "sdid":
    #         return (
    #             pd.DataFrame(
    #                 {
    #                     "features": Y_pre_c_intercept.columns,
    #                     "sdid_weight": np.round(self.hat_omega, 3),
    #                 }
    #             ),
    #             pd.DataFrame(
    #                 {
    #                     "time": Y_pre_c_intercept.index,
    #                     "sdid_weight": np.round(self.hat_lambda, 3),
    #                 }
    #             ),
    #         )
    #     elif model == "sc":
    #         return pd.DataFrame(
    #             {
    #                 "features": self.Y_pre_c.columns,
    #                 "sc_weight": np.round(self.hat_omega_ADH, 3),
    #             }
    #         )
    #     elif model == "ElasticNet":
    #         return pd.DataFrame(
    #             {
    #                 "features": Y_pre_c_intercept.columns,
    #                 "ElasticNet_weight": np.round(self.hat_omega_ElasticNet, 3),
    #             }
    #         )
    #     elif model == "Lasso":
    #         return pd.DataFrame(
    #             {
    #                 "features": Y_pre_c_intercept.columns,
    #                 "Lasso_weight": np.round(self.hat_omega_Lasso, 3),
    #             }
    #         )
    #     elif model == "Ridge":
    #         return pd.DataFrame(
    #             {
    #                 "features": Y_pre_c_intercept.columns,
    #                 "Ridge_weight": np.round(self.hat_omega_Ridge, 3),
    #             }
    #         )
    #     else:
    #         return None


    def target_y(self):
        return self.df.loc[self.pre_term[0] : self.post_term[1], self.treatment].mean(axis=1)

    def sdid_trajectory(self):
        hat_omega = self.hat_omega[:-1]
        Y_c = pd.concat([self.Y_pre_c, self.Y_post_c]) # control units in all periods
        n_units = self.Y_pre_c.shape[1]
        start_w = np.repeat(1 / n_units, n_units)

        _intercept = (start_w - hat_omega) 
        _intercept = _intercept @ self.Y_pre_c.T 
        _intercept = _intercept @ self.hat_lambda

        return Y_c.dot(hat_omega) + _intercept
    
    
    def hat_tau(self, model="sdid"): 
        """
        return ATT, the poential outcome of SDID estimator
        """
        result = pd.DataFrame({"actual_y": self.target_y()})
        post_actural_treat = result.loc[self.post_term[0] :, "actual_y"].mean()

        if model == "sdid":
            result["sdid"] = self.sdid_trajectory()

            pre_sdid = result["sdid"].head(len(self.hat_lambda)) @ self.hat_lambda
            post_sdid = result.loc[self.post_term[0] :, "sdid"].mean()

            pre_treat = (self.Y_pre_t.T @ self.hat_lambda).values[0]
            counterfuctual_post_treat = pre_treat + (post_sdid - pre_sdid)

        elif model == "sc":
            result["sc"] = self.sc_potentical_outcome()
            post_sc = result.loc[self.post_term[0] :, "sc"].mean()
            counterfuctual_post_treat = post_sc

        elif model == "did":
            Y_pre_t = self.Y_pre_t.copy()
            Y_post_t = self.Y_post_t.copy()
            if type(Y_pre_t) != pd.DataFrame:
                Y_pre_t = pd.DataFrame(Y_pre_t)

            if type(Y_post_t) != pd.DataFrame:
                Y_post_t = pd.DataFrame(Y_post_t)

            # actural treat
            post_actural_treat = (
                Y_post_t.mean(axis=1).mean() - Y_pre_t.mean(axis=1).mean()
            )
            counterfuctual_post_treat = (
                self.Y_post_c.mean(axis=1).mean() - self.Y_pre_c.mean(axis=1).mean()
            )

        return post_actural_treat - counterfuctual_post_treat






