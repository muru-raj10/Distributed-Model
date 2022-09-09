#utility functions
import numpy as np
import math
import random
import itertools
import networkx as nx
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from BRVFL_base import BRVFL_cl_f

def initialise_nat_site(middle,inp,init_val=0.):
    #by default initialises to 0 mean, 0 precision
    site_nat={}
    site_nat['r']= np.zeros((middle,inp)) #r_k
    site_nat['Q']= (init_val)*np.eye(middle) #Q_k
    return site_nat

def initialise_nat_site_igamma(alpha1=0,alpha2=0):
    site_nat={}
    site_nat['alpha']=alpha1
    site_nat['beta']=alpha2
    return site_nat

def is_pd(K):
    #checks if matrix K is positive definite
    try:
        np.linalg.cholesky(K)
        return True
    except np.linalg.linalg.LinAlgError:
        return False

def Convert_list_to_dic(lst):
    #converts list output to dict output
    res_dct = {i+1: lst[i] for i in range(len(lst))}
    return res_dct


def invert_gaussian_params(a,B,method='to_natural'):
    """a is the mean parameter or r natural parameter
       B is the covariance matrix or precision matrix Q
       check positive semi-definite before inverting"""
    if method == 'to_natural':
        res = {}
        res['Q'] = np.linalg.inv(B)
        res['r'] = np.matmul(res['Q'],a)
        return res
    else:
        cov = np.linalg.inv(B)
        mu = np.matmul(cov,a)
        return (mu,cov)

def invert_invgamma_params(alpha,beta,method='to_natural'):
    if method == 'to_natural':
        res = {}
        res['beta'] = -beta
        res['alpha'] = -alpha-1
        return res
    else:
        alph = -alpha-1
        bet = -beta
        return (alph,bet)

###################To score data################
def ScoreData(test,mean,var,method='MAP_RE'):
    print(method)
    #test : true values
    #mean : mean of the predictive distribution
    #scores each data point
    scors= []
    if method=='MAP_RE':
        for row in range(len(test)):
            scor = np.sum((test.loc[row].values - mean[row])**2)
            scors.append(scor)
    elif method=='Pred_Var':
        scors = var
    elif method == 'Heuristic':
        for row in range(len(test)):
            scor = np.sum((test.loc[row].values - mean[row])**2)
            scors.append(scor*np.sqrt(var[row]))

    #elif method=='Pred_Prob': #the likelihood
    #    for row in range(len(test)):
    #        prob = 1
    #        for i in range(len(test.loc[row])):
    #            prob *= norm.pdf(test.loc[row].values[i],mean[row][i],var[row])
            #scor = multivariate_normal.pdf(test.loc[row].values,mean[row],var[row]*np.eye(np.shape(mean[row])[0]))
    #        scors.append(prob)

    return scors


def ScoreModel(test,mean,method = 'R2'):
    """
    :return: scores along each dimension or variable
    """
    scors = []
    if method == 'R2':
        emp_mean = np.mean(test)
        cols = test.columns
        for col in range(len(cols)):
            num = np.sum((test[cols[col]].values - np.array(mean)[:,col]) ** 2)
            denom = np.sum((test[cols[col]].values - emp_mean[cols[col]]) ** 2)
            if denom != 0:
                #happens for binary variable where all value is 0
                scors.append(1 - num / denom)

    elif method == 'EVar':
        cols = test.columns
        for col in range(len(cols)):
            num = np.var(test[cols[col]].values - np.array(mean)[:,col])
            denom = np.var(test[cols[col]].values)
            if denom != 0:
                #happens for binary variable where all value is 0
                scors.append(1 - num / denom)

    elif method == 'MSE':
        cols = test.columns
        for col in range(len(cols)):
            num = np.sum((test[cols[col]].values - np.array(mean)[:,col]) ** 2)
            scors.append(num/len(test))

    return scors


#############-------EP functions -------######
def change_in_site(nat_params,cav_nat_params,tilted_nat_params):
    #compute change in each site
    print('computing change in site...')
    change_params = {}
    for k1,v1 in nat_params.items():
        change_params[k1] = tilted_nat_params[k1] - cav_nat_params[k1] - nat_params[k1]

    return change_params

def cavity_at_site(global_nat_params,nat_params,omega=1.0):
    # first cavity parameters at each node is r1,Q1 = r,Q - 0 --> 0, 0.1I as initial site params = 0
    # require condition that cavity inverse covariance matrix be positive definite, else reduce delta at update site
    #for one dimension, inverse covariance be positive.
    print('computing cavity at site...')
    cav_nat_params = {}
    for k1,v1 in nat_params.items():
        cav_nat_params[k1] = global_nat_params[k1] -  omega*nat_params[k1]

    return cav_nat_params

def update_site(change_params,nat_params,delta=1):
    print('updating site parameters...')
    updated_nat_params = {}
    for k1,v1 in nat_params.items():
        updated_nat_params[k1] = nat_params[k1] + delta*change_params[k1]

    return updated_nat_params


def testing_only_site(Normalized_testing_df, labels,global_nat_params,fac=10,me='one'):
    inp = np.shape(Normalized_testing_df)[1]  # the input layer.
    middle = int(fac * inp)

    brvfl_global = BRVFL_cl_f(neurons=middle, random_state=2)
    brvfl_global.init_hidden_matrices(inp)
    brvfl_global.W_o, brvfl_global.W_o_sig = invert_gaussian_params(global_nat_params['r'], global_nat_params['Q'],
                                                                    method='to_mean cov')

    map_est = []
    var_est = []
    if me=='one':
        for row in range(len(Normalized_testing_df)):
            testpoint = (Normalized_testing_df.loc[row].values).reshape(1, -1)
            pred_mean, pred_var = brvfl_global.predict(testpoint)
            map_est.append(pred_mean[0])
            var_est.append(pred_var[0][0])
    else:
        test_d = Normalized_testing_df.values
        pred_mean,pred_var = brvfl_global.predict_mat(test_d)
        var_est = list(pred_var)
        map_est = pred_mean

    scores = ScoreData(Normalized_testing_df, map_est, var_est, method='Pred_Var')  # 'MAP_RE' , 'Heuristic', 'Pred_Var'
    scores2 = ScoreData(Normalized_testing_df, map_est, var_est, method='Heuristic')

    test_auc = roc_auc_score(labels, scores)
    test_auc2 = roc_auc_score(labels, scores2)
    #av_pr = average_precision_score(labels, scores)
    print('Test AUC = {}'.format(test_auc))
    #print('Average Precision = {}'.format(av_pr))

    return (test_auc,test_auc2)



def testing_only_dist(Normalized_testing_df, labels, components,global_nat_params,fac=10,me='one'):
    #separate testing using global parameters in each site. for distributed only
    inp = np.shape(Normalized_testing_df)[1]  # the input layer.
    middle = int(fac * inp)
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

    for site in list(range(1, components + 1)):
        map_est = []
        var_est = []
        if me=='one':
            for row in range(len(Normalized_testing_df)):
                testpoint = (Normalized_testing_df.loc[row].values).reshape(1, -1)
                pred_mean, pred_var = brvfl_global[site].predict(testpoint)
                map_est.append(pred_mean[0])
                var_est.append(pred_var[0][0])
        else:
            test_d = Normalized_testing_df.values
            pred_mean, pred_var = brvfl_global.predict_mat(test_d)
            var_est = list(pred_var)
            map_est = pred_mean

        scores = ScoreData(Normalized_testing_df, map_est, var_est,method='Pred_Var')
        scores2 = ScoreData(Normalized_testing_df, map_est, var_est, method='Heuristic')
        test_auc = roc_auc_score(labels, scores)
        #test_auc2 = roc_auc_score(labels, scores2)
        av_pr = average_precision_score(labels, scores)
        print('Test AUC = {}'.format(test_auc))
        #print('Average Precision = {}'.format(av_pr))
        brvflae_vauc[site] = roc_auc_score(labels, scores)
        brvflae_hauc[site] = roc_auc_score(labels, scores2)

    return brvflae_vauc,brvflae_hauc


def Average_Score(auc_dict):
    mean = np.mean(list(auc_dict.values()))
    sd = np.std(list(auc_dict.values()))
    return (mean*100,sd*100)

###############-EP analysis--##################3
def init_site_sum(nbr_components):
    sumv = {}
    for site in list(range(1, nbr_components + 1)):
        sumv[site]=0
    return sumv

def change_site_per_iter(changes,nbr_iter,nbr_components):
    #changes is a dict. changes[iter][site] = {'r': [array], 'Q':[array]}
    #nbr_iter is number of iterations
    av_r = init_site_sum(nbr_iter)
    av_Q = init_site_sum(nbr_iter)
    means_r = []
    means_Q = []
    for k1, v1 in changes.items():
        for k2, v2 in changes[k1].items():
            av_r[k1] += changes[k1][k2]['r']/nbr_components
            av_Q[k1] += changes[k1][k2]['Q']/nbr_components
        #means_r.append(np.mean(np.sqrt(np.sum((av_r[k1]) ** 2, axis=0)))) #should we take mean or sum?
        #means_Q.append(np.sqrt(np.sum(np.sum((av_Q[k1]) ** 2, axis=0)))) #Frobenius norm of matrix
        means_r.append(np.mean(abs(av_r[k1]))) #should we take mean or sum? #mean as each are independent
        means_Q.append(np.mean(abs(av_Q[k1]))) #Frobenius norm of matrix

    return (means_r,means_Q)

def avchange_site_per_iter(changes,nbr_components):
    #changes is a dict. changes[iter][site] = {'r': [array], 'Q':[array]}
    #nbr_iter is number of iterations
    #for absolute difference in iterations
    means_r = []
    means_Q = []
    for k1, v1 in changes.items():
        av_r = 0
        av_Q = 0
        for k2, v2 in changes[k1].items():
            #av_r[k1] += np.mean(np.sqrt(np.sum((changes[k1][k2]['r']) ** 2, axis=0)))/nbr_components
            #av_Q[k1] += np.sqrt(np.sum(np.sum((changes[k1][k2]['Q']) ** 2, axis=0)))/nbr_components
            av_r += np.mean(abs(changes[k1][k2]['r']))
            av_Q += np.mean(abs(changes[k1][k2]['Q']))
        #means_r.append(np.mean(np.sqrt(np.sum((av_r[k1]) ** 2, axis=0)))) #should we take mean or sum?
        #means_Q.append(np.sqrt(np.sum(np.sum((av_Q[k1]) ** 2, axis=0)))) #Frobenius norm of matrix
        means_r.append(av_r/nbr_components) #should we take mean or sum? #mean as each are independent
        means_Q.append(av_Q/nbr_components) #Frobenius norm of matrix

    return (means_r,means_Q)

def avglobal_change_site_per_iter(changes,one_node_global,nbr_components):
    #changes is a dict. changes[iter][site] = {'r': [array], 'Q':[array]}
    #nbr_iter is number of iterations
    #av_r = init_site_sum(nbr_iter)
    #av_Q = init_site_sum(nbr_iter)0
    means_r = []
    means_Q = []
    denom_r = np.mean(np.sqrt(np.sum((one_node_global['r']) ** 2, axis=0)))
    denom_Q = np.sqrt(np.sum(np.sum((one_node_global['Q']) ** 2, axis=0)))
    for k1, v1 in changes.items(): #iterations
        ae_r = 0
        ae_Q = 0
        for k2, v2 in changes[k1].items(): #sites
            ae_r += np.mean(np.sqrt(np.sum((changes[k1][k2]['r']-one_node_global['r']) ** 2, axis=0)))/(nbr_components*denom_r)
            ae_Q += np.sqrt(np.sum(np.sum((changes[k1][k2]['Q']-one_node_global['Q']) ** 2, axis=0)))/(nbr_components*denom_Q)
            #av_r[k1+1] += np.mean(np.sqrt(np.sum((changes[k1][k2]['r']-one_node_global['r']) ** 2, axis=0)))/(nbr_components*denom_r)
            #av_Q[k1+1] += np.sqrt(np.sum(np.sum((changes[k1][k2]['Q']-one_node_global['Q']) ** 2, axis=0)))/(nbr_components*denom_Q)
        means_r.append(ae_r) #should we take mean or sum? #mean as each are independent
        means_Q.append(ae_Q) #Frobenius norm of matrix

    return (means_r,means_Q)

def change_global(changes_iter,one_node_global):
    #difference for central site implementation

    g_r_means =[]
    g_Q_means = []
    denom_r = np.mean(np.sqrt(np.sum((one_node_global['r']) ** 2, axis=0)))
    denom_Q = np.sqrt(np.sum(np.sum((one_node_global['Q']) ** 2, axis=0)))
    for k1, v1 in changes_iter.items():
        av_r = changes_iter[k1]['r']-one_node_global['r']
        av_Q = changes_iter[k1]['Q']-one_node_global['Q']
        g_r_means.append(np.mean(np.sqrt(np.sum((av_r) ** 2, axis=0)))/denom_r) #should we take mean or sum? #mean as each are independent
        g_Q_means.append(np.sqrt(np.sum(np.sum((av_Q) ** 2, axis=0)))/denom_Q) #Frobenius norm of matrix
        #g_r_means.append(np.mean(abs(av_r[k1]))) #should we take mean or sum? #mean as each are independent
        #g_Q_means.append(np.mean(abs(av_Q[k1]))) #Frobenius norm of matrix

    return (g_r_means,g_Q_means)





#########################################
def Get_Neighs(sites,kap=0.5,seed=2):
    G = nx.Graph()
    neighs = {}
    for i in range(1, sites + 1):
        neighs[i] = set()

    linkss = list(itertools.combinations(range(1,sites+1), 2))
    edges = int(kap * sites * (sites - 1) / 2) #nbr of 'two way' connections
    print(edges)
    print('kappa:{}'.format(2*edges/(sites*(sites-1))))
    if edges < sites:
        print(edges)
        print('kappa too small')
        return 0
    random.seed(seed)
    alllink = set(random.sample(linkss, edges))
    for elt in alllink:
        neighs[elt[0]].add(elt[1])
        neighs[elt[1]].add(elt[0])

    G.add_edges_from(alllink)

    if len(list(nx.connected_components(G)))!=1:
        print(len(list(nx.connected_components(G))))
        print('Faulty network components. Not all connected')
        return 0
    #shortestpaths = dict(nx.all_pairs_shortest_path(G))  #shortestpaths[1][4] gives shortesst path from site 1 to site 4
    length = dict(nx.all_pairs_shortest_path_length(G)) #length[1][4] gives shortest path length from site 1 to 4
    maxhop = 0
    for i in range(1, sites + 1):
        for j in range(1, sites + 1):
            hop = length[i][j]
            if hop>maxhop:
                maxhop=hop
    print('Max Hop for a site is {}'.format(maxhop))

    return neighs

def Get_Graph(sites,kap=0.5,seed=2):
    G = nx.Graph()
    linkss = list(itertools.combinations(range(1,sites+1), 2))
    edges = int(kap * sites * (sites - 1) / 2) #nbr of 'two way' connections
    random.seed(seed)
    alllink = set(random.sample(linkss, edges))

    G.add_edges_from(alllink)
    #import matplotlib.pyplot as plt
    #nx.draw(G)
    return G

#import matplotlib.pyplot as plt

#G = Get_Graph(50,kap=0.2)
#plt.clf()
#nx.draw(G,pos=nx.shell_layout(G),node_size=100)
#plt.savefig('50_02.png')
