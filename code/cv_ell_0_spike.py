import os, sys, time, pickle, glob,scipy
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
# %matplotlib inline
# sns.set(color_codes=True)
from itertools import product
# from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import pprint
# import allensdk.brain_observatory.stimulus_info as stim_info
from itertools import combinations 
from numba import jit,njit, prange
from sklearn.metrics import adjusted_rand_score,mean_squared_error
# argparser for running code on command line
import argparse

sys.path.append('./OASIS/')
import oasis
from oasis import oasisAR1, oasisAR2
import neo
from neo.core import SpikeTrain
from quantities import s
import elephant
# helper functions 

@jit(nopython=True,parallel=False, fastmath=False)
def Dcost_func(y,gamma = 0.9):
    """
    for a given sequence, compute D(y_1,y_n) for 
    """
    assert len(y)>=1, 'must have at least 1 data point'
    num_c = 0.
    denom_c = 0.
    cost = 0.
    for t,y_t in enumerate(y):
        num_c += y_t*np.power(gamma,t)
        denom_c += np.power(gamma,2*t)
    c_hat = num_c/denom_c
    for t,y_t in enumerate(y):
        cost += np.power(y_t-np.power(gamma,t)*c_hat,2)
    cost*=0.5
    return([c_hat,cost])
	

# @jit(nopython=True,parallel=True, fastmath=True)
def backtrack_spike(back_vec):
    """
    Take in a list and backtrack where we estimated a spike for ell_0
    """
    # you start from the end
    estimated_spike = []
    counter = len(back_vec)-1
    estimated_spike.append(back_vec[counter])
    while counter!=0:
        counter = int(back_vec[counter])
        estimated_spike.append(back_vec[counter])
    # because we are backtracking so we need to reverse the result
    estimated_spike = estimated_spike[::-1][1:]
    return(estimated_spike)


def data_generation(T, theta, gamma, sigma, random_seed = 1234):
        """
        Generate data using an AR-1 process - same as paper's simulation condition 
	    Input:
	    T: timestep to simulate
	    theta: firing rate
	    gamma: decay rate
	    sigma: noise parameter
	    """
	    
        np.random.seed(random_seed)
        s_t = np.random.poisson(lam = theta, size=T)
        c_0 = 1.0
        c_list = [c_0]
        F_list = [c_0+np.random.normal(loc=0,scale=sigma)]
        for i in range(1,T):
            curr_c = gamma*c_list[i-1]+s_t[i]
            c_list.append(curr_c)
            F_list.append(curr_c+np.random.normal(loc=0,scale=sigma))
        return([s_t,c_list,F_list])

def dp_ell_0_active_sol(y, alpha = 0.01,gamma = 0.9):
    
    T = len(y)
    backtrack = np.zeros(T)
    F_val = np.zeros(T)
    F_val[0] = -alpha
    c_cp_hat = np.zeros(T)
    active_set = set()
    # dp loop
    

    for t in range(1,T):
        active_set.add(t-1)
        active_set_copy = active_set.copy()
        temp_dict = dict()
        for s in active_set_copy: 
#             print('diff', t-len(active_set_copy))
            curr_y = y[s:t]
            _, curr_cost = Dcost_func(curr_y,gamma)
            temp_dict[s] =  F_val[s]+alpha+curr_cost
        #
        curr_min = min(temp_dict, key=temp_dict.get)
#         c_cp_hat[t] = curr_c_hat
        backtrack[t] = curr_min
        F_val[t] = temp_dict[curr_min]
        
        # we should prune the active set until we get a solution
        for s in active_set_copy: 
            if (temp_dict[s]-alpha)>=F_val[t]:
                active_set.remove(s)
        
    # post-process to construct calcium fit
    # TODO: we will wrap this up into a class later LOL
    
    spike_fit = sorted(backtrack_spike(backtrack))
    spike_fit = [int(x) for x in spike_fit]
    c_hat_list = np.zeros(T)
    
    for i in range(len(spike_fit)):
        if (i<(len(spike_fit)-1)):
            c_hat_list[spike_fit[i]], _ = Dcost_func(y[spike_fit[i]:spike_fit[i+1]],gamma=gamma)
        else:
            c_hat_list[spike_fit[i]], _ = Dcost_func(y[spike_fit[i]:],gamma=gamma)
            
    for t in range(T):
        if (t not in spike_fit and  t>0):
            c_hat_list[t] = c_hat_list[t-1]*gamma
            
            
    return([F_val,spike_fit,c_hat_list])



from quantities import s

def cv_select_lambda(y,s_vec,lambda_seq,L_min_seq,gamma = 0.96):
    
    train_y = y[0::2]
    test_y = y[1::2]
    mse_c_l0 = []
    mse_c_l1 = []
    result_dict = dict()
    
    
    l_0_vr_list = []
    l_1_vr_list = []
    l_0_vp_list = []
    l_1_vp_list = []
    
    for lam in lambda_seq:
        print('lamba = ',lam)
        c_t_l1 , s_t_l1 = oasisAR1(train_y, g = gamma,lam =lam, s_min=0)
        _, s_t_l0,c_t_l0 = dp_ell_0_active_sol(train_y, alpha = lam,gamma = gamma)
        
        test_c_t_l0 = [(a + b) / 2 for a, b in \
                       zip(c_t_l0[0:-1], c_t_l0[1:])]
        test_c_t_l1 = [(a + b) / 2 for a, b in \
                       zip(c_t_l1[0:-1], c_t_l1[1:])]
        
        mse_c_l0.append(mean_squared_error(test_c_t_l0,test_y[0:-1]))
        mse_c_l1.append(mean_squared_error(test_c_t_l1,test_y[0:-1]))
        
        
        c_t_l1 , s_t_l1 = oasisAR1(y, g = gamma,lam =lam, s_min=0)
        _, s_t_l0,c_t_l0 = dp_ell_0_active_sol(y, alpha = lam,gamma = gamma)
        
        t_scale = len(y)
        t_vec = np.arange(0,t_scale)
        l_0_spike = s_t_l0

        l_0_spike_collection.append(t_vec[l_0_spike])
        l_1_spike_collection.append([t_vec[np.where(s_t_l1>L_min)] for L_min in L_min_seq])


        spike_truth = SpikeTrain(np.where(s_vec>0)*s, t_stop=len(s_vec))
        l_1_spike_list = [SpikeTrain(np.where(s_t_l1>L_min)*s, t_stop=len(s_vec))\
                          for L_min in L_min_seq]
        l_0_spike = SpikeTrain(l_0_spike*s, t_stop=len(s_vec))
        
        # compute post-processing estimators
        l_1_vr = [elephant.spike_train_dissimilarity.van_rossum_dist([l_1_spike,\
                     spike_truth],tau=2.*s)[0,1] for l_1_spike in l_1_spike_list]
        l_1_vp = [elephant.spike_train_dissimilarity.victor_purpura_dist([l_1_spike,\
                      spike_truth])[0,1] for l_1_spike in  l_1_spike_list]
        
        l_0_vr = elephant.spike_train_dissimilarity.van_rossum_dist([l_0_spike,spike_truth],tau=2*s)[0,1]        
        l_0_vp = elephant.spike_train_dissimilarity.victor_purpura_dist([l_0_spike,spike_truth])[0,1]
        
        l_0_vr_list.append(l_0_vr)
        l_1_vr_list.append(l_1_vr)
        l_0_vp_list.append(l_0_vp)
        l_1_vp_list.append(l_1_vp)

    # store results
    result_dict['lambda_seq'] = lambda_seq
    result_dict['L_min_seq'] = L_min_seq # :(
    result_dict['mse_c_l0'] = mse_c_l0 
    result_dict['mse_c_l1'] = mse_c_l1 
    result_dict['l_0_vr_list'] = l_0_vr_list
    result_dict['l_1_vr_list'] = l_1_vr_list
    result_dict['l_0_vp_list'] = l_0_vp_list
    result_dict['l_1_vp_list'] = l_1_vp_list
    result_dict['l_1_spike_collection'] = l_1_spike_collection
    result_dict['l_0_spike_collection'] = l_0_spike_collection
    return(result_dict)



initial_seed = 1234
np.random.seed(initial_seed)


output_dir = '/home/students/yiqunc/stat572/output/simulation/'

spike_len = 1000
spike_gamma = 0.96
spike_theta = 0.01
spike_sigma = 0.15

parse_output_name = 'more_lambda_cv_result_' + 'T_' + str(spike_len) + '_sigma_' + \
str(spike_sigma)  + '_theta_' + str(spike_theta) + '_gamma_'+ str(spike_gamma) + '.pkl'


repitition = 50
result_list = []

L_seq = [0, 0.125, 0.25, 0.375, 0.5]
lambda_choice = np.logspace(-2, 2, num=10)

for i in range(repitition):
    
    print('iteration',i)
    current_seed = int(initial_seed+i)

    spike_data = data_generation(T=spike_len, \
    theta = spike_theta, gamma = spike_gamma, sigma = spike_sigma, random_seed = current_seed)
    # observed fluoresence
    obs_y = np.array(spike_data[2])
    # observed calcium
    obs_c = np.array(spike_data[1])
    # observed spike
    obs_s = np.array(spike_data[0])

    
    test_result = cv_select_lambda(y= obs_y,s_vec=obs_s,\
                       lambda_seq = lambda_choice ,L_min_seq = L_seq,\
                       gamma =spike_sigma)

    result_list.append(test_result)
    
with open(output_dir+parse_output_name, 'wb') as handle:
    pickle.dump(result_list, handle, protocol=pickle.HIGHEST_PROTOCOL)







