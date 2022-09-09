import pandas as pd
import numpy as np
import time
from utility_func import *

from loop_unsw import EP_w_cen,EP_hypers_cen, EP_w_cen_onebyone
from dist_unsw import EP_w,EP_hypers,EP_w_hops,EP_hypers_hops

from loop_unsw import Run_nbrcomp

sites = 10
kappa = 0.8

Normalized_training_df,Normalized_testing_df,labels,components = Run_nbrcomp(1)
gams_s = EP_hypers_cen(Normalized_training_df, components, 'prior',mloops=8,fac=10)
sig_s = EP_hypers_cen(Normalized_training_df, components, 'lhood',mloops=8,fac=10)
global_nat_params_s,changes_s,global_params_iters_s,test_auc_s = EP_w_cen(Normalized_training_df,Normalized_testing_df,labels,components,
                                                                          sig_s,gams_s,fac=10,mloops=3,testing_d=False)


#central (kappa=1)
Normalized_training_df,Normalized_testing_df,labels,components = Run_nbrcomp(sites)
gams_c = EP_hypers_cen(Normalized_training_df, components, 'prior',mloops=8,fac=10)
sig_c = EP_hypers_cen(Normalized_training_df, components, 'lhood',mloops=8,fac=10)
max_loop_c = 4
global_nat_params_c,changes_c,global_params_iters_c,test_auc_c = EP_w_cen(Normalized_training_df,Normalized_testing_df,labels,components,
                                                                          sig_c,gams_c,fac=10,mloops=max_loop_c,delt=1,testing_d=False)



#for distributed
neigh_sites = Get_Neighs(components,kappa)
gams = EP_hypers(Normalized_training_df, components, 'prior',neigh_sites,mloops=8,fac=10)
sig = EP_hypers(Normalized_training_df, components, 'lhood',neigh_sites,mloops=8,fac=10)
global_nat_params, changes, global_params_iters, brvflae_vauc,brvflae_hauc = EP_w(Normalized_training_df, Normalized_testing_df, labels, components,
                                                                                  sig, gams, neigh_sites, fac=10, delt=1, mloops=max_loop_c,testing_d=False)
