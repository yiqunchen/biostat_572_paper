import os, sys, time, pickle, glob,scipy
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
# %matplotlib inline
# sns.set(color_codes=True)
from itertools import product
#from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import pprint
#import allensdk.brain_observatory.stimulus_info as stim_info
from itertools import combinations 
from numba import jit,njit, prange
from sklearn.metrics import adjusted_rand_score,mean_squared_error
# argparser for running code on command line
import argparse


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
	

#@jit(nopython=True,parallel=True, fastmath=True)
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


def dp_ell_0_sol(y, alpha = 0.01,gamma = 0.9):
    # this uses a for loop to 
    T = len(y)
    backtrack = np.zeros(T)
    F_val = np.zeros(T)
    F_val[0] = -alpha
    c_cp_hat = np.zeros(T)
    
    # dp loop
    for t in range(1,T):
        curr_list = np.zeros(t)
        c_hat_list = np.zeros(t)
        for s in range(0,t):
            curr_y = y[s:t]
            _, curr_cost = Dcost_func(curr_y,gamma=gamma)
            curr_list[s] = F_val[s]+alpha+curr_cost
        curr_min = np.argmin(curr_list)
        backtrack[t] = curr_min
        F_val[t] = curr_list[curr_min]
    

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


def dp_ell_0_active_sol(y, alpha = 0.01,gamma = 0.9):
    
    T = len(y)
    backtrack = np.zeros(T)
    F_val = np.zeros(T)
    F_val[0] = -alpha
    c_cp_hat = np.zeros(T)
    active_set = set()
    # dp loop
    
    # dp loop without the pruning
    for t in range(1,T):
        curr_list = np.zeros(t)
        c_hat_list = np.zeros(t)
        for s in range(0,t):
            curr_y = y[s:t]
            _, curr_cost = Dcost_func(curr_y,gamma=gamma)
            curr_list[s] = F_val[s]+alpha+curr_cost
        curr_min = np.argmin(curr_list)
        backtrack[t] = curr_min
        F_val[t] = curr_list[curr_min]
        
    for t in range(1,T):
        active_set.add(t-1)
        active_set_copy = active_set.copy()
        temp_dict = dict()
        for s in active_set_copy: 
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
            if (F_val[s]+curr_cost)>=F_val[t]:
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




### simulation case
parser = argparse.ArgumentParser()
parser.add_argument('-T',"--time",\
 help = 'Specify the length of observations', type = float)
# 
parser.add_argument('-S',"--sigma",\
 help = 'Specify the sd of the observations', type = float)

parser.add_argument('-mu',"--theta",\
 help = 'Specify the mean of poisson process', type = float)

parser.add_argument('-G',"--gamma",\
 help = 'Specify the decay of calcium process', type = float)

args = parser.parse_args()

spike_len = int(args.time/1)
spike_sigma = args.sigma
spike_theta = args.theta/1
spike_gamma = args.gamma

initial_seed = 1234
np.random.seed(initial_seed)


output_dir = '/home/students/yiqunc/stat572/output/simulation/'

parse_output_name = 'time_result' + '_T_' + str(spike_len) + '_sigma_' + \
str(spike_sigma)  + '_theta_' + str(spike_theta) + '_gamma_'+ str(spike_gamma) + '.pkl'



repitition = 1
active_timing = []
normal_timing = []

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

    t_1 = time.time()
    F_val_active,backtrack_active,c_cp_hat_active = dp_ell_0_active_sol(obs_y,\
 alpha = 0.1,gamma = 0.998)
    t_2 = time.time()
    print(t_2-t_1,'no')
    t_3 = time.time()
    F_val_normal ,backtrack_normal ,c_cp_hat_normal  = dp_ell_0_sol(obs_y, \
   alpha = 0.1,gamma = 0.998)
    t_4 = time.time()

    active_timing.append(t_2-t_1)
    normal_timing.append(t_4-t_3)

timing_result = {'active_timing':active_timing,'normal_timing':normal_timing}

with open(output_dir+parse_output_name, 'wb') as handle:
    pickle.dump(timing_result, handle, protocol=pickle.HIGHEST_PROTOCOL)












