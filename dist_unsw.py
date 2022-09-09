import pandas as pd
import numpy as np
import random
import math
import time
from joblib import Parallel, delayed
import multiprocessing
from utility_func import *
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from BRVFL_base import BRVFL_cl_f

random.seed(30)
np.random.seed(30)
direc = 'UNSW-NB15 - CSV Files/'
#direc = 'UNSW-NB15 - CSV Files/'
f1 = 'UNSW_NB15_training-set.csv'
f2 = 'UNSW_NB15_testing-set.csv'
f1 = pd.read_csv(direc + f1)
f2 = pd.read_csv(direc + f2)


def Run_nbrcomp(components, datapts=0):
    print(components)
    print('unsw-nb15')

    from sklearn.feature_extraction import DictVectorizer
    def encode_onehot(df, cols):
        """
        One-hot encoding is applied to columns specified in a pandas DataFrame.
        @param df pandas DataFrame
        @param cols a list of columns to encode
        @return a DataFrame with one-hot encoding
        """
        vec = DictVectorizer()

        vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
        vec_data.columns = vec.get_feature_names()
        vec_data.index = df.index

        df = df.drop(cols, axis=1)
        df = df.join(vec_data)
        return df

    features = ['dur', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
                'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss',
                'sintpkt', 'dintpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
                'tcprtt', 'synack', 'ackdat', 'smeansz', 'dmeansz', 'trans_depth',
                'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
                'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
                'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
                'ct_srv_dst', 'is_sm_ips_ports']

    train_enc = encode_onehot(f1, ['service', 'state'])
    test_enc = encode_onehot(f2, ['service', 'state'])

    def Check_same_values(train, test, colms=['service', 'state']):
        for col in colms:
            not_in_test = set(train[col]) - set(train[col]).intersection(set(test[col]))
            not_in_train = set(test[col]) - set(train[col]).intersection(set(test[col]))
            print(not_in_test)
            print(not_in_train)
            if not_in_train:
                for item in not_in_train:
                    col_name = col + '={}'.format(item)
                    train_enc[col_name] = 0
            if not_in_test:
                for item in not_in_test:
                    col_name = col + '={}'.format(item)
                    test_enc[col_name] = 0

    Check_same_values(f1, f2, colms=['service', 'state'])

    binary_cols = ['is_sm_ips_ports', 'is_ftp_login']
    nominal_cols = []
    nominal_cols = list(train_enc.columns[list(train_enc.columns).index('label') + 1:])
    binary_cols.extend(nominal_cols)
    del nominal_cols
    print(len(binary_cols))

    numerical_cols = list(train_enc.columns[:list(train_enc.columns).index('label') - 1])
    numerical_cols.remove('id')
    # numerical_cols.remove('srcip')
    # numerical_cols.remove('sport')
    # numerical_cols.remove('dstip')
    # numerical_cols.remove('dstp')
    numerical_cols.remove('is_sm_ips_ports')
    numerical_cols.remove('is_ftp_login')
    numerical_cols.remove('proto')
    # numerical_cols.remove('stime')
    # numerical_cols.remove('ltime')
    print(len(numerical_cols))

    norm_train_df = train_enc[train_enc['label'] == 0].reset_index(drop=True)
    print(len(norm_train_df))

    # cannot perform PCA map on binary data because the data is split
    # can't normalize with respect to all data. Normalizin with respect to site data makes them unidentifiable with each other.

    for col in numerical_cols:
        if len(norm_train_df[col].unique()) == 1:
            print('{} has only 1 unique value, {}'.format(col, norm_train_df[col].unique()))
            # numerical_cols.remove(col)

    if datapts != 0:
        tot_samps = components * datapts
        norm_train_df = norm_train_df.sample(n=tot_samps, random_state=5).reset_index(drop=True)

    from sklearn.utils import shuffle
    def SplitData(norm_train_df, components=2, rand=2):
        data = shuffle(norm_train_df, random_state=rand)
        site_data = {}
        length = math.floor(len(data) / components)
        for i in range(components):
            site_data[i + 1] = data[i * length:(i + 1) * length]

        return site_data

    from sklearn import mixture
    def SplitDataGMM(norm_train_df, cols, components=2):
        gmm = mixture.GaussianMixture(n_components=components)
        labels = gmm.fit_predict(norm_train_df[cols])
        labels = labels + 1
        norm_train_df['gmml'] = labels
        site_data = {}
        for i in set(labels):
            site_data[i] = norm_train_df[norm_train_df['gmml'] == i]
        return site_data

    all_cols = numerical_cols.copy()
    all_cols.extend(binary_cols)
    # site_data = SplitDataGMM(norm_train_df,all_cols,components)
    site_data = SplitData(norm_train_df, components)

    for k, v in site_data.items():
        site_data[k] = site_data[k].reset_index(drop=True)

    mean = 0
    for k, v in site_data.items():
        mean += site_data[k][numerical_cols].mean(axis=0)

    # combine mean at main node. Same as mean of all data
    means = mean / components

    # meanz = norm_train_df[numerical_cols].mean(axis=0)

    # normalize numerical cols to range (0,1) by mapping mean of the feature value to 0.5
    def f(x, val=0):
        return (1 - np.exp(-x * val)) / (1 + np.exp(-x * val)) - 0.5

    from scipy import optimize
    def get_k_val(means, cols=numerical_cols):
        k_val = pd.DataFrame(columns=['k_val'], index=cols)

        for col in cols:
            val = means[col]
            try:
                root = optimize.brentq(f, 0, 100000, val)
                # print('{} : {}'.format(col,root))
                k_val['k_val'].loc[col] = root
            except:
                print(col)
        return k_val

    k_val = get_k_val(means, cols=numerical_cols)  # transmit k_val back to site for normalizing

    # normalize numerical column in each site
    Normalized_training_df = {}
    for k, v in site_data.items():
        Normalized_training_df[k] = pd.DataFrame(columns=numerical_cols)
        for col in numerical_cols:
            Normalized_training_df[k][col] = f(site_data[k][col], k_val['k_val'].loc[col]) + 0.5
        Normalized_training_df[k][binary_cols] = site_data[k][binary_cols]

    Normalized_testing_df = pd.DataFrame(columns=numerical_cols)
    for col in numerical_cols:
        Normalized_testing_df[col] = f(test_enc[col], k_val['k_val'].loc[col]) + 0.5

    Normalized_testing_df[binary_cols] = test_enc[binary_cols]
    labels = test_enc['label']

    print(len(Normalized_training_df[1]) * components)
    print(labels.value_counts())

    return (Normalized_training_df, Normalized_testing_df, labels, components)


#######################################################################################
def EP_hypers(Normalized_training_df, components, meth, neigh_sites, mloops=8, fac=10,omega=1.0):
    inp = np.shape(Normalized_training_df[1])[1]  # the input layer.
    middle = int(fac * inp)
    print('middle:{}'.format(middle))

    sites_data = {}
    for k, v in Normalized_training_df.items():
        sites_data[k] = Normalized_training_df[k].values

    sites_nat_params = {}
    for site, v in sites_data.items():  # 0 prior
        sites_nat_params[site] = initialise_nat_site_igamma()

    sites_changes_params = {}
    for site, v in sites_data.items():
        sites_changes_params[site] = initialise_nat_site_igamma()

    sites_delta = 2 + np.zeros(components)  # each site will have a delta for update. initialise to 2

    ########################## ----------------EP functions-------------------#######################################
    # functions which contain global variables defined above

    def tilted_invgam(site_data, cav_nat_params, method='lhood',omega=omega):
        # At each site perform bayesian inference and compute tilted distribution
        # Not using eta right now

        print('computing tilted distribution...')
        print('alpha:{}'.format(cav_nat_params['alpha']))
        print('beta:{}'.format(cav_nat_params['beta']))
        params = invert_invgamma_params(cav_nat_params['alpha'], cav_nat_params['beta'], method='to_shape,rate')
        # print('prior: {}'.format(params[1]))
        # initial sigma is the mode of the inverse gamma dist
        print(params[1] / (1 + params[0]))
        if method == 'lhood':
            rvfl_f = BRVFL_cl_f(neurons=middle, random_state=2, sigma=params[1] / (1 + params[0]),
                                Mean=np.zeros((middle, inp)), Gamma=np.eye(middle),omega=omega)
        else:
            rvfl_f = BRVFL_cl_f(neurons=middle, random_state=2, Mean=np.zeros((middle, inp)),
                                Gamma=(params[1] / (1 + params[0])) * np.eye(middle),omega=omega)
        rvfl_f.fit(site_data, site_data)  # to get the mean W_0
        rvfl_f.fit_hp(site_data, site_data)  # to get tot_sq_res
        if method == 'lhood':
            new_alpha = params[0] + rvfl_f.nbr_data / 2  # gelman 3rd edition bayesian data analysis pg 43
            new_beta = params[1] + rvfl_f.tot_sq_res / 2  #..scaled inv chi^2(a,b) = IG(a/2,ab/2)
        else:
            new_alpha = params[0] + middle / 2
            new_beta = params[1] + np.sum(
                np.sum((rvfl_f.W_o) ** 2, axis=0)) / 2  # np.mean(np.sum((rvfl_f.W_o) ** 2, axis=0))

        tilted_nat_params = invert_invgamma_params(new_alpha, new_beta, method='to_natural')

        return tilted_nat_params

    def delta_x_change(deltas, all_change_params, neighs):
        """ Sum over sites,, (sites_delta,sites_changes_params)
        """
        delxchanges = initialise_nat_site_igamma()
        for site in neighs:
            for k1, v1 in all_change_params[site].items():
                delxchanges[k1] += deltas * all_change_params[site][k1]

        return delxchanges

    def SiteUpdate(site_delta, site):
        # operation at site. receive delta from central
        sites_nat_params[site] = update_site(sites_changes_params[site], sites_nat_params[site], delta=site_delta)
        return sites_nat_params[site]

    def Site_Operation(global_nat_params, sites_nat_params, site, method='lhood',omega=omega):
        # operation at site. compute cavity, tilted and changes. receive global nat params from central
        print(omega)
        sites_cavity_params = cavity_at_site(global_nat_params, sites_nat_params[site], omega=omega)
        sites_tilted_params = tilted_invgam(sites_data[site], sites_cavity_params, method=method,omega=omega)
        sites_changes_params = change_in_site(sites_nat_params[site], sites_cavity_params, sites_tilted_params)
        return sites_changes_params

    def Central_Operation(sites_changes_params,neigh_sites, site,omega=omega):
        # receive site changes from sites. outputs global params and delta
        # delta needs to be reduced if site cavity dist are not pos definite
        print('central operation')
        neighs = neigh_sites[site]
        neis_and_self = neighs.copy()
        neis_and_self.add(site)
        delts = 1
        central_sites_nat_params = {}
        print('site: {}'.format(site))
        site_cavity = initialise_nat_site_igamma()
        while (site_cavity['alpha'] >= -1) or (site_cavity['beta'] >= 0):
            print('not right!!')
            delts = 0.5 * delts
            central_sites_nat_params = update_site(sites_changes_params[site], sites_nat_params[site],delta=delts)
            new_global_params = update_site(delta_x_change(delts, sites_changes_params,neis_and_self), global_nat_params[site])
            site_cavity = cavity_at_site(new_global_params, central_sites_nat_params,omega=omega)
        print('new delts:{}'.format(delts))
        return new_global_params, delts

    loop = 1
    max_loop = mloops
    # meth = 'lhood'  #change here
    # meth = 'prior'

    if meth == 'lhood':
        global_nat_params = {}
        for site, v in sites_data.items():  # 0 prior
            global_nat_params[site] = initialise_nat_site_igamma(-100,-1)
        # (-1-10**-5,-10**-5)  #(-100, -1) -> initial value of sigma^2 =  0.01
    else:
        global_nat_params = {}
        for site, v in sites_data.items():  # 0 prior
            global_nat_params[site] = initialise_nat_site_igamma(-10,-1000)
        # (-1, -1)->initial value of gamma-1 =1, (-10, -1000) for 100.  #(-1-10**-5,-10**-5)
        # alpha = -r -1 , beta = -Q  ---> mode = beta/(alpha+1) --> -Q/(-r)
        # alpha>0,beta>0 ---> r<-1, Q<0

    t0 = time.time()
    while loop <= max_loop:
        print('Iteration : {}'.format(loop))
        site_results = Parallel(n_jobs=4)(
            delayed(SiteUpdate)(sites_delta[site-1], site) for site in list(range(1, components + 1)))
        sites_nat_params = Convert_list_to_dic(site_results)
        # site_params_naturals[loop]= sites_nat_params.copy() #for analysis later
        for site in list(range(1, components + 1)):  # nbr of components = nbr of sites
            print('site: {}'.format(site))
            sites_changes_params[site] = Site_Operation(global_nat_params[site], sites_nat_params, site, method=meth)
        # change_results= Parallel(n_jobs=2)(delayed(Site_Operation)(global_nat_params,sites_nat_params, site,eta=1,method = meth) for site in list(range(1,components+1)))
        # sites_changes_params = Convert_list_to_dic(change_results)
        for site in list(range(1, components + 1)):
            global_nat_params[site], sites_delta[site-1] = Central_Operation(sites_changes_params, neigh_sites, site)
        print('all sites complete')
        loop += 1

    t1 = time.time()
    print('training time: {} minutes'.format((t1 - t0) / 60))

    ##################################################################################################
    # using global params
    modes_p={}
    for site in list(range(1, components + 1)):
        shape, rate = invert_invgamma_params(global_nat_params[site]['alpha'],
                                             global_nat_params[site]['beta'], method='to_shape,rate')

        print('Mode: {}'.format(rate / (shape + 1)))
        modes_p[site] = rate / (shape + 1)
    return modes_p


##################################################################################################
def EP_w(Normalized_training_df, Normalized_testing_df, labels, components, sig, gam, neigh_sites,
         fac=10, delt=1, mloops=10,omega=1.0,testing_d=True):
    # fac = val to x input layer, delt = delta, mloops=max ep loops, testing_d=perf testing if true
    # middle = 500 #try 500 or 1000  #actually the middle layer
    inp = np.shape(Normalized_training_df[1])[1]  # the input layer.
    middle = int(fac * inp)  # try 500 or 1000  #actually the middle layer
    print(middle)
    # middle = 4*inp
    sig = sig
    gams = gam

    sites_data = {}
    for k, v in Normalized_training_df.items():
        sites_data[k] = Normalized_training_df[k].values

    global_nat_params = {}
    for site, v in sites_data.items():  # 0 prior
        global_nat_params[site] = initialise_nat_site(middle, inp, init_val=1 / gams[site])  # 1.0/100  #init_val=1/0.019501351863718694

    sites_nat_params = {}
    for site, v in sites_data.items():  # 0 prior
        sites_nat_params[site] = initialise_nat_site(middle, inp)

    sites_changes_params = {}
    for site, v in sites_data.items():
        sites_changes_params[site] = initialise_nat_site(middle, inp)

    sites_delta = 2 * delt + np.zeros(components)  # each site will have a delta for update. initialise to 2

    ########################## ----------------EP functions-------------------#######################################
    # functions which contain global variables defined above

    def tilted_brvfl(site_data, cav_nat_params, site, method='closed',omega=omega):
        """
        At each site perform bayesian inference and compute tilted distribution
        Not using eta right now
        """
        print('computing tilted distribution...')
        params = invert_gaussian_params(cav_nat_params['r'], cav_nat_params['Q'], method='to_mean,cov')
        # print('prior: {}'.format(params[1]))
        if method == 'closed':  # this is closed form posterior includes the normalizing term
            rvfl_f = BRVFL_cl_f(neurons=middle, random_state=2, sigma=sig[site], Mean=params[0],Gamma=params[1],omega=omega)
            # sigma = 10**-2, 6.73170063337287e-05 #0.00010507046296819523
            rvfl_f.fit(site_data, site_data)
            tilted_nat_params = invert_gaussian_params(rvfl_f.W_o, rvfl_f.W_o_sig, method='to_natural')

        return tilted_nat_params

    def delta_x_change(deltas, all_change_params,neighs):
        #Sum over sites,, (sites_delta,sites_changes_params)
        delxchanges = initialise_nat_site(middle, inp)
        for site in neighs:
            for k1, v1 in all_change_params[site].items():
                delxchanges[k1] += deltas * all_change_params[site][k1]

        return delxchanges

    def SiteUpdate(site_delta, site):
        # operation at site. receive delta from central
        sites_nat_params[site] = update_site(sites_changes_params[site], sites_nat_params[site], delta=site_delta)
        return sites_nat_params[site]

    def Site_Operation(global_nat_params, sites_nat_params, site, method='closed',omega=omega):
        # operation at site. compute cavity, tilted and changes. receive global nat params from central
        print(omega)
        sites_cavity_params = cavity_at_site(global_nat_params, sites_nat_params[site], omega=omega)
        sites_tilted_params = tilted_brvfl(sites_data[site], sites_cavity_params, site, method=method,omega=omega)
        sites_changes_params = change_in_site(sites_nat_params[site], sites_cavity_params, sites_tilted_params)
        return sites_changes_params

    def Central_Operation(sites_changes_params,neigh_sites, site, delta=delt,omega=omega):
        # receive site changes from sites. outputs global params and delta
        # delta needs to be reduced if site cavity dist are not pos definite
        print('central operation')
        neighs = neigh_sites[site]
        neis_and_self = neighs.copy()
        neis_and_self.add(site)
        delts = 2 * delta
        print(delts)
        central_sites_nat_params = {}
        print('site: {}'.format(site))
        site_cavity = initialise_nat_site(middle, inp)
        while not is_pd(site_cavity['Q']):
            print('not positive definite!')
            delts = 0.5 * delts  # currently using only one value of delta (delta=1 works!)
            central_sites_nat_params = update_site(sites_changes_params[site], sites_nat_params[site], delta= delts)
            new_global_params = update_site(delta_x_change(delts, sites_changes_params, neis_and_self), global_nat_params[site])
            site_cavity = cavity_at_site(new_global_params, central_sites_nat_params, omega=omega)
        print('new delts:{}'.format(delts))
        return new_global_params, delts

    loop = 1
    max_loop = mloops
    changes = {}
    site_params_naturals = {}  # not yet implemented .. for analysis later
    global_params_iters = {}
    meth = 'closed'  # change here
    global_params_iters[0] = global_nat_params.copy()
    t0 = time.time()
    while loop <= max_loop:
        #om = omega/(loop)
        om = omega
        print('Iteration : {}'.format(loop))
        site_results = Parallel(n_jobs=2)(
            delayed(SiteUpdate)(sites_delta[site-1], site) for site in list(range(1, components + 1)))  # chagne here
        sites_nat_params = Convert_list_to_dic(site_results)
        # site_params_naturals[loop]= sites_nat_params.copy() #this is after the update before global update. no point analysing
        for site in list(range(1, components + 1)):  # nbr of components = nbr of sites
            print('site: {}'.format(site))
            sites_changes_params[site] = Site_Operation(global_nat_params[site], sites_nat_params, site, method=meth,omega=om)
        changes[loop] = sites_changes_params.copy()  # for analysis later

        for site in list(range(1, components + 1)):
            global_nat_params[site], sites_delta[site-1] = Central_Operation(sites_changes_params, neigh_sites, site,omega=om)
        # change_results= Parallel(n_jobs=2)(delayed(Site_Operation)(global_nat_params,sites_nat_params, site,eta=1,method = meth) for site in list(range(1,components+1)))
        # sites_changes_params = Convert_list_to_dic(change_results)
        print('all sites complete')
        global_params_iters[loop] = global_nat_params.copy()
        loop += 1

    t1 = time.time()
    print('training time: {} minutes'.format((t1 - t0) / 60))

    ##################################################################################################
    # using global params
    brvfl_global={}
    for site in list(range(1, components + 1)):
        brvfl_global[site] = BRVFL_cl_f(neurons=middle, random_state=2)
        brvfl_global[site].init_hidden_matrices(inp)
        brvfl_global[site].W_o, brvfl_global[site].W_o_sig = invert_gaussian_params(global_nat_params[site]['r'],
                                                                                    global_nat_params[site]['Q'],
                                                                                    method='to_mean cov')
    ##################################################################################################3
    # testing
    brvflae_vauc = {}
    brvflae_hauc = {}
    t0 = time.time()
    if testing_d:
        for site in list(range(1, components + 1)):
            map_est = []
            var_est = []
            for row in range(len(Normalized_testing_df)):
                testpoint = (Normalized_testing_df.loc[row].values).reshape(1, -1)
                pred_mean, pred_var = brvfl_global[site].predict(testpoint)
                map_est.append(pred_mean[0])
                var_est.append(pred_var[0][0])
            scores = ScoreData(Normalized_testing_df, map_est, var_est,method='Pred_Var')
            scores2 = ScoreData(Normalized_testing_df, map_est, var_est, method='Heuristic')
            test_auc = roc_auc_score(labels, scores)
            #test_auc2 = roc_auc_score(labels, scores2)
            av_pr = average_precision_score(labels, scores)
            print('Test AUC = {}'.format(test_auc))
            #print('Average Precision = {}'.format(av_pr))
            brvflae_vauc[site] = roc_auc_score(labels, scores)
            brvflae_hauc[site] = roc_auc_score(labels, scores2)

    t1 = time.time()
    print('testing time: {} minutes'.format((t1 - t0) / 60))


    print('done')
    return global_nat_params, changes, global_params_iters, brvflae_vauc, brvflae_hauc
    # return (scores,scores2)
    # return (brvflae_vauc,brvflae_hauc)
    # return (map_est, var_est)






#######################################################################################
def EP_hypers_hops(Normalized_training_df, components, meth, kap=1, mloops=8, fac=10,omega=1.0):
    inp = np.shape(Normalized_training_df[1])[1]  # the input layer.
    middle = int(fac * inp)
    print('middle:{}'.format(middle))

    sites_data = {}
    for k, v in Normalized_training_df.items():
        sites_data[k] = Normalized_training_df[k].values

    sites_nat_params = {}
    for site, v in sites_data.items():  # 0 prior
        sites_nat_params[site] = initialise_nat_site_igamma()

    sites_changes_params = {}
    for site, v in sites_data.items():
        sites_changes_params[site] = initialise_nat_site_igamma()

    sites_delta = 2 + np.zeros(components)  # each site will have a delta for update. initialise to 2

    ########################## ----------------EP functions-------------------#######################################
    # functions which contain global variables defined above

    def tilted_invgam(site_data, cav_nat_params, method='lhood',omega=omega):
        # At each site perform bayesian inference and compute tilted distribution
        # Not using eta right now

        print('computing tilted distribution...')
        print('alpha:{}'.format(cav_nat_params['alpha']))
        print('beta:{}'.format(cav_nat_params['beta']))
        params = invert_invgamma_params(cav_nat_params['alpha'], cav_nat_params['beta'], method='to_shape,rate')
        # print('prior: {}'.format(params[1]))
        # initial sigma is the mode of the inverse gamma dist
        print(params[1] / (1 + params[0]))
        if method == 'lhood':
            rvfl_f = BRVFL_cl_f(neurons=middle, random_state=2, sigma=params[1] / (1 + params[0]),
                                Mean=np.zeros((middle, inp)), Gamma=np.eye(middle),omega=omega)
        else:
            rvfl_f = BRVFL_cl_f(neurons=middle, random_state=2, Mean=np.zeros((middle, inp)),
                                Gamma=(params[1] / (1 + params[0])) * np.eye(middle),omega=omega)
        rvfl_f.fit(site_data, site_data)  # to get the mean W_0
        rvfl_f.fit_hp(site_data, site_data)  # to get tot_sq_res
        if method == 'lhood':
            new_alpha = params[0] + rvfl_f.nbr_data / 2  # gelman 3rd edition bayesian data analysis pg 43
            new_beta = params[1] + rvfl_f.tot_sq_res / 2  #..scaled inv chi^2(a,b) = IG(a/2,ab/2)
        else:
            new_alpha = params[0] + middle / 2
            new_beta = params[1] + np.sum(
                np.sum((rvfl_f.W_o) ** 2, axis=0)) / 2  # np.mean(np.sum((rvfl_f.W_o) ** 2, axis=0))

        tilted_nat_params = invert_invgamma_params(new_alpha, new_beta, method='to_natural')

        return tilted_nat_params

    def delta_x_change(deltas, all_change_params, neighs):
        """ Sum over sites,, (sites_delta,sites_changes_params)
        """
        delxchanges = initialise_nat_site_igamma()
        for site in neighs:
            for k1, v1 in all_change_params[site].items():
                delxchanges[k1] += deltas * all_change_params[site][k1]

        return delxchanges

    def SiteUpdate(site_delta, site):
        # operation at site. receive delta from central
        sites_nat_params[site] = update_site(sites_changes_params[site], sites_nat_params[site], delta=site_delta)
        return sites_nat_params[site]

    def Site_Operation(global_nat_params, sites_nat_params, site, method='lhood',omega=omega):
        # operation at site. compute cavity, tilted and changes. receive global nat params from central
        print(omega)
        sites_cavity_params = cavity_at_site(global_nat_params, sites_nat_params[site], omega=omega)
        sites_tilted_params = tilted_invgam(sites_data[site], sites_cavity_params, method=method,omega=omega)
        sites_changes_params = change_in_site(sites_nat_params[site], sites_cavity_params, sites_tilted_params)
        return sites_changes_params

    def Central_Operation(sites_changes_params,neigh_gr,loop, site,omega=omega):
        # receive site changes from sites. outputs global params and delta
        # delta needs to be reduced if site cavity dist are not pos definite
        print('central operation')
        neis_and_self = set(nx.single_source_shortest_path(neigh_gr, site, loop).keys())
        delts = 1
        central_sites_nat_params = {}
        print('site: {}'.format(site))
        site_cavity = initialise_nat_site_igamma()
        while (site_cavity['alpha'] >= -1) or (site_cavity['beta'] >= 0):
            print('not right!!')
            delts = 0.5 * delts
            central_sites_nat_params = update_site(sites_changes_params[site], sites_nat_params[site],delta=delts)
            new_global_params = update_site(delta_x_change(delts, sites_changes_params,neis_and_self), global_nat_params[site])
            site_cavity = cavity_at_site(new_global_params, central_sites_nat_params,omega=omega)
        print('new delts:{}'.format(delts))
        return new_global_params, delts

    loop = 1
    max_loop = mloops
    # meth = 'lhood'  #change here
    # meth = 'prior'

    if meth == 'lhood':
        global_nat_params = {}
        for site, v in sites_data.items():  # 0 prior
            global_nat_params[site] = initialise_nat_site_igamma(-100,-1)
        # (-1-10**-5,-10**-5)  #(-100, -1) -> initial value of sigma^2 =  0.01
    else:
        global_nat_params = {}
        for site, v in sites_data.items():  # 0 prior
            global_nat_params[site] = initialise_nat_site_igamma(-10,-1000)
        # (-1, -1)->initial value of gamma-1 =1, (-10, -1000) for 100.  #(-1-10**-5,-10**-5)
        # alpha = -r -1 , beta = -Q  ---> mode = beta/(alpha+1) --> -Q/(-r)
        # alpha>0,beta>0 ---> r<-1, Q<0
    neigh_gr = Get_Graph(components, kap)
    t0 = time.time()
    while loop <= max_loop:
        print('Iteration : {}'.format(loop))
        site_results = Parallel(n_jobs=4)(
            delayed(SiteUpdate)(sites_delta[site-1], site) for site in list(range(1, components + 1)))
        sites_nat_params = Convert_list_to_dic(site_results)
        # site_params_naturals[loop]= sites_nat_params.copy() #for analysis later
        for site in list(range(1, components + 1)):  # nbr of components = nbr of sites
            print('site: {}'.format(site))
            sites_changes_params[site] = Site_Operation(global_nat_params[site], sites_nat_params, site, method=meth)
        # change_results= Parallel(n_jobs=2)(delayed(Site_Operation)(global_nat_params,sites_nat_params, site,eta=1,method = meth) for site in list(range(1,components+1)))
        # sites_changes_params = Convert_list_to_dic(change_results)
        for site in list(range(1, components + 1)):
            global_nat_params[site], sites_delta[site-1] = Central_Operation(sites_changes_params, neigh_gr,loop, site)
        print('all sites complete')
        loop += 1

    t1 = time.time()
    print('training time: {} minutes'.format((t1 - t0) / 60))

    ##################################################################################################
    # using global params
    modes_p={}
    for site in list(range(1, components + 1)):
        shape, rate = invert_invgamma_params(global_nat_params[site]['alpha'],
                                             global_nat_params[site]['beta'], method='to_shape,rate')

        print('Mode: {}'.format(rate / (shape + 1)))
        modes_p[site] = rate / (shape + 1)
    return modes_p


##################################################################################################
def EP_w_hops(Normalized_training_df, Normalized_testing_df, labels, components, sig, gam,
              kap=1,fac=10, delt=1, mloops=10,omega=1.0,testing_d=True):
    # fac = val to x input layer, delt = delta, mloops=max ep loops, testing_d=perf testing if true
    # middle = 500 #try 500 or 1000  #actually the middle layer
    inp = np.shape(Normalized_training_df[1])[1]  # the input layer.
    middle = int(fac * inp)  # try 500 or 1000  #actually the middle layer
    print(middle)
    # middle = 4*inp

    sites_data = {}
    for k, v in Normalized_training_df.items():
        sites_data[k] = Normalized_training_df[k].values

    global_nat_params = {}
    for site, v in sites_data.items():  # 0 prior
        global_nat_params[site] = initialise_nat_site(middle, inp, init_val=1 / gams[site])  # 1.0/100  #init_val=1/0.019501351863718694

    sites_nat_params = {}
    for site, v in sites_data.items():  # 0 prior
        sites_nat_params[site] = initialise_nat_site(middle, inp)

    sites_changes_params = {}
    for site, v in sites_data.items():
        sites_changes_params[site] = initialise_nat_site(middle, inp)

    sites_delta = 2 * delt + np.zeros(components)  # each site will have a delta for update. initialise to 2

    ########################## ----------------EP functions-------------------#######################################
    # functions which contain global variables defined above

    def tilted_brvfl(site_data, cav_nat_params, site, method='closed',omega=omega):
        """
        At each site perform bayesian inference and compute tilted distribution
        Not using eta right now
        """
        print('computing tilted distribution...')
        params = invert_gaussian_params(cav_nat_params['r'], cav_nat_params['Q'], method='to_mean,cov')
        # print('prior: {}'.format(params[1]))
        if method == 'closed':  # this is closed form posterior includes the normalizing term
            rvfl_f = BRVFL_cl_f(neurons=middle, random_state=2, sigma=sig[site], Mean=params[0],Gamma=params[1],omega=omega)
            # sigma = 10**-2, 6.73170063337287e-05 #0.00010507046296819523
            rvfl_f.fit(site_data, site_data)
            tilted_nat_params = invert_gaussian_params(rvfl_f.W_o, rvfl_f.W_o_sig, method='to_natural')

        return tilted_nat_params

    def delta_x_change(deltas, all_change_params,neighs):
        #Sum over sites,, (sites_delta,sites_changes_params)
        delxchanges = initialise_nat_site(middle, inp)
        for site in neighs:
            for k1, v1 in all_change_params[site].items():
                delxchanges[k1] += deltas * all_change_params[site][k1]

        return delxchanges

    def SiteUpdate(site_delta, site):
        # operation at site. receive delta from central
        sites_nat_params[site] = update_site(sites_changes_params[site], sites_nat_params[site], delta=site_delta)
        return sites_nat_params[site]

    def Site_Operation(global_nat_params, sites_nat_params, site, method='closed',omega=omega):
        # operation at site. compute cavity, tilted and changes. receive global nat params from central
        print(omega)
        sites_cavity_params = cavity_at_site(global_nat_params, sites_nat_params[site], omega=omega)
        sites_tilted_params = tilted_brvfl(sites_data[site], sites_cavity_params, site, method=method,omega=omega)
        sites_changes_params = change_in_site(sites_nat_params[site], sites_cavity_params, sites_tilted_params)
        return sites_changes_params

    def Central_Operation(sites_changes_params, neigh_gr,loop,site, delta=delt,omega=omega):
        # receive site changes from sites. outputs global params and delta
        # delta needs to be reduced if site cavity dist are not pos definite
        #takes next hops neighbours in each loop. problem: should not update the other sites. does not work as intended.
        print('central operation')
        neis_and_self = set(nx.single_source_shortest_path(neigh_gr, site, loop).keys())
        delts = 2 * delta
        print(delts)
        central_sites_nat_params = {}
        print('site: {}'.format(site))
        site_cavity = initialise_nat_site(middle, inp)
        while not is_pd(site_cavity['Q']):
            print('not positive definite!')
            delts = 0.5 * delts  # currently using only one value of delta (delta=1 works!)
            central_sites_nat_params = update_site(sites_changes_params[site], sites_nat_params[site], delta= delts)
            new_global_params = update_site(delta_x_change(delts, sites_changes_params, neis_and_self), global_nat_params[site])
            site_cavity = cavity_at_site(new_global_params, central_sites_nat_params, omega=omega)
        print('new delts:{}'.format(delts))
        return new_global_params, delts

    loop = 1
    max_loop = mloops
    changes = {}
    site_params_naturals = {}  # not yet implemented .. for analysis later
    global_params_iters = {}
    meth = 'closed'  # change here
    global_params_iters[0] = global_nat_params.copy()
    neigh_gr = Get_Graph(components,kap)
    t0 = time.time()
    while loop <= max_loop:
        #om = omega/(loop)
        om = omega
        print('Iteration : {}'.format(loop))
        site_results = Parallel(n_jobs=2)(
            delayed(SiteUpdate)(sites_delta[site-1], site) for site in list(range(1, components + 1)))  # chagne here
        sites_nat_params = Convert_list_to_dic(site_results)
        # site_params_naturals[loop]= sites_nat_params.copy() #this is after the update before global update. no point analysing
        for site in list(range(1, components + 1)):  # nbr of components = nbr of sites
            print('site: {}'.format(site))
            sites_changes_params[site] = Site_Operation(global_nat_params[site], sites_nat_params, site, method=meth,omega=om)
        changes[loop] = sites_changes_params.copy()  # for analysis later

        for site in list(range(1, components + 1)):
            global_nat_params[site], sites_delta[site-1] = Central_Operation(sites_changes_params, neigh_gr,loop-1,site,omega=om)
        # change_results= Parallel(n_jobs=2)(delayed(Site_Operation)(global_nat_params,sites_nat_params, site,eta=1,method = meth) for site in list(range(1,components+1)))
        # sites_changes_params = Convert_list_to_dic(change_results)
        print('all sites complete')
        global_params_iters[loop] = global_nat_params.copy()
        loop += 1

    t1 = time.time()
    print('training time: {} minutes'.format((t1 - t0) / 60))

    ##################################################################################################
    # using global params
    brvfl_global={}
    for site in list(range(1, components + 1)):
        brvfl_global[site] = BRVFL_cl_f(neurons=middle, random_state=2)
        brvfl_global[site].init_hidden_matrices(inp)
        brvfl_global[site].W_o, brvfl_global[site].W_o_sig = invert_gaussian_params(global_nat_params[site]['r'],
                                                                                    global_nat_params[site]['Q'],
                                                                                    method='to_mean cov')
    ##################################################################################################3
    # testing
    brvflae_vauc = {}
    brvflae_hauc = {}
    t0 = time.time()
    if testing_d:
        for site in list(range(1, components + 1)):
            map_est = []
            var_est = []
            for row in range(len(Normalized_testing_df)):
                testpoint = (Normalized_testing_df.loc[row].values).reshape(1, -1)
                pred_mean, pred_var = brvfl_global[site].predict(testpoint)
                map_est.append(pred_mean[0])
                var_est.append(pred_var[0][0])
            scores = ScoreData(Normalized_testing_df, map_est, var_est,method='Pred_Var')
            scores2 = ScoreData(Normalized_testing_df, map_est, var_est, method='Heuristic')
            test_auc = roc_auc_score(labels, scores)
            #test_auc2 = roc_auc_score(labels, scores2)
            av_pr = average_precision_score(labels, scores)
            print('Test AUC = {}'.format(test_auc))
            #print('Average Precision = {}'.format(av_pr))
            brvflae_vauc[site] = roc_auc_score(labels, scores)
            brvflae_hauc[site] = roc_auc_score(labels, scores2)

    t1 = time.time()
    print('testing time: {} minutes'.format((t1 - t0) / 60))


    print('done')
    return global_nat_params, changes, global_params_iters, brvflae_vauc, brvflae_hauc
    # return (scores,scores2)
    # return (brvflae_vauc,brvflae_hauc)
    # return (map_est, var_est)
