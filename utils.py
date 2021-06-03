#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import collections
from collections import defaultdict
from scipy.special import softmax 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import scale
import pickle
import time
import os
import os.path as osp
import sys
from sklearn.metrics.pairwise import euclidean_distances as ecdist
import multiprocessing
from functools import partial
import requests, zipfile, io
import random
from sklearn.cluster.k_means_ import _init_centroids


def normalizefea(X):
    """
    X: stacked feature data matrix
    return: L2 normalized data matrix
    """
    feanorm = np.maximum(1e-14,np.sum(X**2,axis=1))
    X_out = X/(feanorm[:,None]**0.5)
    return X_out

def km_le(X,M):
    
    """
    X: stacked point matrix
    M: stacked center matrix
    return: point assignment (clustering labels) to their closest centers
    
    """
    e_dist = ecdist(X,M)          
    l = e_dist.argmin(axis=1)
        
    return l


'''
class MiniBatchKMeans: implement the classical mini-batch k-means algorithm from scratch
'''
class MiniBatchKMeans:
    n_clusters = 2
    dim_n = 2
    init_centroids = None
    init_labels = None
    final_centroids = None
    final_labels = None
    
    def __init__(self, name, n_samples = 400, seed = 1):
        ## generate synthetic data or load real dataset
        if name == 'Synthetic-equal':
            centers = [(1, 1), (2.1, 1), (1, 5), (2.1, 5)]
            self.data_org, self.gender = make_blobs(n_samples=n_samples, n_features=self.dim_n, cluster_std=0.1,
                      centers=centers, shuffle=False, random_state=1)

            index = n_samples // 2
            self.gender[0:index] = 0
            self.gender[index:n_samples] = 1
            
            data = self.data_org

        elif name == 'Synthetic2-equal':
            sample_list = [200,200]
            centers = [(1, 1), (2.1, 3.5)]
            self.data_org, self.gender = make_blobs(n_samples=sample_list, n_features=self.dim_n, cluster_std=0.3,                                        centers=centers, shuffle=False, random_state=seed)

            index = sample_list[0]
            self.gender[0:index] = 0
            self.gender[index:] = 1
            
            data = self.data_org
    
        elif name == 'Synthetic-unequal':
            sample_list = [150,150,50,50]
            centers = [(1, 1), (2.1, 1), (1, 3.5), (2.1, 3.5)]
            self.data_org, self.gender = make_blobs(n_samples=sample_list, n_features=self.dim_n, cluster_std=0.13,                                    centers=centers, shuffle=False, random_state=seed)

            index = sample_list[0]+sample_list[1]

            self.gender[0:index] = 0
            self.gender[index:] = 1
            
            data = self.data_org
            
        elif name == 'Synthetic2-unequal': 
            sample_list = [300,100]
            centers = [(1, 1), (2.1, 3.5)]
            self.data_org, self.gender = make_blobs(n_samples=sample_list, n_features=self.dim_n, cluster_std=0.3,                                    centers=centers, shuffle=False, random_state=seed)
            index = sample_list[0]
            self.gender[0:index] = 0
            self.gender[index:] = 1
            
            data = self.data_org

        elif name == 'Adult':
            data_dir = "data/"
            savepath = osp.join(data_dir, name+'.npz')

            datas = np.load(savepath)
            X_org = datas['X_org']
            demograph = datas['demograph']
            K = datas['K'].item()
            
            n_samples = len(demograph)
            self.data_org, self.gender, K = X_org, demograph, K
            self.n_clusters = n_clu
            self.dim_n = 5
            self.n_demoGroups = len(list(set(self.gender)))

            data = scale(self.data_org, axis = 0)
            data = normalizefea(data)
            
        elif name == 'Bank':
            data_dir = "data/"
            savepath = osp.join(data_dir, name+'.npz')
            
            datas = np.load(savepath)
            X_org = datas['X_org']
            demograph = datas['demograph']
            K = datas['K'].item()
            
            n_samples = len(demograph)
            self.data_org, self.gender, K = X_org, demograph, K
            self.n_clusters = n_clu
            self.dim_n = 6
            self.n_demoGroups = len(list(set(self.gender)))

            data = scale(self.data_org, axis = 0)
            data = normalizefea(data)

        
        if name in ['Adult', 'Bank']: 
            np.random.seed(seed)
            random.seed(seed)
            
            n_samples = 5000
            numbers = np.random.choice(len(self.gender), size=n_samples, replace=False)
            
            self.gender = self.gender[numbers]
            self.point_mat = np.zeros([n_samples, self.dim_n + 1])
            self.point_mat[:, :self.dim_n] = data[numbers, :self.dim_n]
            self.point_mat[:, self.dim_n] = self.gender.astype(int)

        else:
            self.gender = self.gender[:n_samples]
            self.point_mat = np.zeros([n_samples, self.dim_n + 1])
            self.point_mat[:, :self.dim_n] = data[:n_samples, 0:self.dim_n]
            self.point_mat[:, self.dim_n] = self.gender.astype(int)
        
        self.data_name = name
        self.n_samples = n_samples
        self.max_iters = self.n_samples//10
    
    def initial_clustering(self, seed = 1):
        np.random.seed(seed)
        self.init_centroids = self.point_mat[np.random.randint(self.n_samples, size = self.n_clusters), 0:self.dim_n]*1.0
        self.init_labels = self.regular_assignment(self.point_mat[:, 0:self.dim_n], self.init_centroids)
        
    ## define the regular assignment function: assign to the closest point
    def regular_assignment(self, pt, centroids):
        arr_dist = np.zeros([len(pt), self.n_clusters])
        for i in range(self.n_clusters):
            arr_dist[:, i] =  np.linalg.norm(pt - centroids[i, :], axis = 1)
        label = np.argmin(arr_dist, axis = 1)
        return label

    def plot_demo(self):
        colmap_demo = {0: 'k', 1: 'm', 2: 'g', 3:'y', 4:'DarkGreen', 5:'r', 6:'b', 7:'c', 8: 'LightBlue', 9:"Orange"}
        colors = list(map(lambda x: colmap_demo[x], self.gender))
        plt.scatter(self.point_mat[:, 0], self.point_mat[:, 1], color=colors, alpha=0.5, edgecolor='k')
        plt.title("Demograpic composition: black->males, purple->females")
        plt.show()
    
    def plot_initClu(self):
        colmap_clu = {0: 'r', 1: 'b', 2: 'g', 3:'y', 4:'DarkGreen', 5:'k', 6:'m', 7:'c', 8: 'LightBlue', 9:"Orange"}
        colors = list(map(lambda x: colmap_clu[x], self.init_labels))
        
        fig2 = plt.figure(figsize=(5, 5))
        plt.scatter(self.point_mat[:, 0], self.point_mat[:, 1], color=colors, alpha=0.3, edgecolor='k')
        for i in range(self.n_clusters):
            plt.scatter(self.init_centroids[i, 0], self.init_centroids[i, 1], color=colmap_clu[i])
        plt.title("Initial clustering using initial centroids: red vs blue")
        plt.show()
    
    def update_objective_f1(self, centroids, arr_label):
        sum_dist = 0
        for i in range(self.n_clusters):
            idx = np.where(arr_label == i)[0]
            data_c1 = self.point_mat[idx, 0:self.dim_n]
            sum_dist += np.sum(np.linalg.norm(data_c1 - centroids[i, :], axis = 1)**2)
        return sum_dist/len(self.point_mat)
    
    def plot_FinalClu(self):
        colmap_clu = {0: 'r', 1: 'b', 2: 'g', 3:'y', 4:'DarkGreen', 5:'k', 6:'m', 7:'c', 8: 'LightBlue', 9:"Orange"}
        colors = list(map(lambda x: colmap_clu[x], self.final_labels))

        fig3 = plt.figure(figsize=(5, 5))
        plt.scatter(self.point_mat[:, 0], self.point_mat[:, 1], color=colors, alpha=0.3, edgecolor='k')
        for i in range(self.n_clusters):
            plt.scatter(self.final_centroids[i, 0], self.final_centroids[i, 1], color=colmap_clu[i])
        plt.title("Final clustering using final centroids: red vs blue")
        plt.show()

    def miniBatchKmeans(self, max_iters = 400, batch_size = 10):
        centroids = self.init_centroids
        
        arr_label = np.zeros(self.n_samples) # self.init_labels
        cluster_size = np.zeros(self.n_clusters)
        
        arr_f1 = np.zeros(max_iters)
        iters = 0
        while iters < max_iters: # or len(np.where(arr_label == 0)[0]) > 10
            batch_index = np.random.choice(self.n_samples, size=batch_size, replace=False)
            batch_matrix = self.point_mat[batch_index, 0:self.dim_n]
            batch_labels = self.regular_assignment(batch_matrix, centroids)
            
            ## record old labels and assign new labels
            old_label = arr_label[batch_index]
            arr_label[batch_index] = batch_labels
            
            for i in range(batch_size):
                label = batch_labels[i]
                oldlabel= int(old_label[i])
                
                if oldlabel == 0:
                    cluster_size[label] += 1
                elif oldlabel != label:
                    cluster_size[oldlabel] -= 1
                    cluster_size[label] += 1
                
                if oldlabel != label:
                    ita = 1.0/cluster_size[label] 
                    centroids[label] = centroids[label]*(1.0 - ita) + batch_matrix[i, :]*ita
                    
            updated_labs = self.regular_assignment(self.point_mat[:, 0:self.dim_n], centroids)
            arr_f1[iters] = self.update_objective_f1(centroids, updated_labs)
            iters += 1
        
        self.final_centroids = centroids
        self.final_labels = updated_labs
        
        # plot 
        plt.plot(arr_f1, 'k.')
        plt.xlabel("Iterations")
        plt.ylabel("Clustering cost")
        plt.show()
        print("Final clustering cost", arr_f1[-1])


class MiniBatchFairKMeans:
    
    n_clusters = 2
    dim_n = 2
    n_demoGroups = 2
    n_init = 10
    
    init_provided = False
    f1_list = None
    f2_list = None
    
    ## parameters of PF-SAGD algorithm
    num_iter = 0
    max_iter = 1000
    max_len_pareto_front = 1500
    
    stepsize = 1.0
    
    point_per_iteration = 3
    num_steps_per_point = 1
    
    dense_threshold = 0
    
    def __init__(self, name, n_samples = 400, seed = 1, n_clu = 10):
        if name == 'Synthetic-equal':
            centers = [(1, 1), (2.1, 1), (1, 5), (2.1, 5)]
            self.data_org, self.gender = make_blobs(n_samples=n_samples, n_features=self.dim_n, cluster_std=0.1,
                      centers=centers, shuffle=False, random_state=1)

            index = n_samples // 2
            self.gender[0:index] = 0
            self.gender[index:n_samples] = 1
            
            data = self.data_org

        elif name == 'Synthetic2-equal':
            sample_list = [200,200]
            centers = [(1, 1), (2.1, 3.5)]
            self.data_org, self.gender = make_blobs(n_samples=sample_list, n_features=self.dim_n, cluster_std=0.3,                                        centers=centers, shuffle=False, random_state=seed)

            index = sample_list[0]
            self.gender[0:index] = 0
            self.gender[index:] = 1
            
            data = self.data_org
    
        elif name == 'Synthetic-unequal':
            sample_list = [150,150,50,50]
            centers = [(1, 1), (2.1, 1), (1, 3.5), (2.1, 3.5)]
            self.data_org, self.gender = make_blobs(n_samples=sample_list, n_features=self.dim_n, cluster_std=0.13,                                    centers=centers, shuffle=False, random_state=seed)

            index = sample_list[0]+sample_list[1]

            self.gender[0:index] = 0
            self.gender[index:] = 1
            
            data = self.data_org
            
        elif name == 'Synthetic2-unequal': 
            sample_list = [300,100]
            centers = [(1, 1), (2.1, 3.5)]
            self.data_org, self.gender = make_blobs(n_samples=sample_list, n_features=self.dim_n, cluster_std=0.3,                                    centers=centers, shuffle=False, random_state=seed)
            index = sample_list[0]
            self.gender[0:index] = 0
            self.gender[index:] = 1
            
            data = self.data_org

        elif name == 'Adult':
            data_dir = "data/"
            savepath = osp.join(data_dir, name+'.npz')

            datas = np.load(savepath)
            X_org = datas['X_org']
            demograph = datas['demograph']
            K = datas['K'].item()
            
            n_samples = len(demograph)
            self.data_org, self.gender, K = X_org, demograph, K
            self.n_clusters = n_clu
            self.dim_n = 5
            self.n_demoGroups = len(list(set(self.gender)))

            data = scale(self.data_org, axis = 0)
            data = normalizefea(data)
            
        elif name == 'Bank':
            data_dir = "data/"
            savepath = osp.join(data_dir, name+'.npz')
            
            datas = np.load(savepath)
            X_org = datas['X_org']
            demograph = datas['demograph']
            K = datas['K'].item()
            
            n_samples = len(demograph)
            self.data_org, self.gender, K = X_org, demograph, K
            self.n_clusters = n_clu
            self.dim_n = 6
            self.n_demoGroups = len(list(set(self.gender)))

            data = scale(self.data_org, axis = 0)
            data = normalizefea(data)

        
        if name in ['Adult', 'Bank']: 
            np.random.seed(seed)
            random.seed(seed)
            
            n_samples = 5000
            numbers = np.random.choice(len(self.gender), size=n_samples, replace=False)
            
            self.gender = self.gender[numbers]
            self.point_mat = np.zeros([n_samples, self.dim_n + 1])
            self.point_mat[:, :self.dim_n] = data[numbers, :self.dim_n]
            self.point_mat[:, self.dim_n] = self.gender.astype(int)

        else:
            self.gender = self.gender[:n_samples]
            self.point_mat = np.zeros([n_samples, self.dim_n + 1])
            self.point_mat[:, :self.dim_n] = data[:n_samples, 0:self.dim_n]
            self.point_mat[:, self.dim_n] = self.gender.astype(int)
    
        self.data_name = name
        self.n_samples = n_samples
        self.max_iters = self.n_samples
        
    ## compute distance of the set of point to the set of centers
    def distance(self, pt, centroids):
        arr_dist = np.zeros([len(pt), self.n_clusters])
        for i in range(self.n_clusters):
            arr_dist[:, i] =  np.linalg.norm(pt - centroids[i, :], axis = 1)
        return arr_dist     
    
    ## compute center according to a specific label
    def update_centers(self, arr_label):
        centroids = np.zeros([self.n_clusters, self.dim_n])
        for i in range(self.n_clusters):
            index_k = np.where(arr_label == i)[0]
            if len(index_k) == 0:
                continue
            centroids[i, :] = np.mean(self.point_mat[index_k, 0:self.dim_n], axis=0)
        return centroids
    
    ## randomly generate num sets of labels
    def init_labels_centers(self, seed = 1):
        np.random.seed(seed)
        random.seed(seed)
        self.arr_init_labels = np.zeros([self.n_init, self.n_samples])*(-1)
        self.arr_init_centroids = np.zeros([self.n_init, self.n_clusters, self.dim_n])
        rlen = self.n_init // 2
        for i in range(rlen):
            if self.data_name in ['Adult', 'Bank']:
                p = np.random.randn(self.n_clusters)
                prob = softmax(p)
                init_labels = np.random.choice(self.n_clusters, self.n_samples, p=prob)
            else:
                p1, p2 = np.random.rand(1)[0], np.random.rand(1)[0]
                init_labels = np.append(np.random.choice(self.n_clusters, 300, p=[p1, 1-p1]), np.random.choice(self.n_clusters, 100, p=[p2, 1-p2]))
            
            init_centroids = self.update_centers(init_labels)
            self.arr_init_labels[i, :] = init_labels
            self.arr_init_centroids[i, :, :] = init_centroids
        
        for i in range(rlen, self.n_init):
            M =_init_centroids(self.point_mat[:, :self.dim_n], self.n_clusters, init='k-means++')
            l = km_le(self.point_mat[:, :self.dim_n], M)
            self.arr_init_labels[i, :] = l
            self.arr_init_centroids[i, :, :] = M
    
    def plot_demo(self, arr_labels):
        count_demo = np.zeros([self.n_clusters, self.n_demoGroups])
        for k in range(self.n_clusters):
            for j in range(self.n_demoGroups):
                count_demo[k, j] = len(np.where((self.gender == j) & (arr_labels == k))[0]) #zeros in cluster 1
         
        balance_1 = np.min(count_demo[0, :])/np.max(count_demo[0, :])
        balance_2 = np.min(count_demo[1, :])/np.max(count_demo[1, :])

        print ("Balance", np.min([balance_1, balance_2]))

        labels = ['C1', 'C2']
        men_means = count_demo[:, 0]
        women_means = count_demo[:, 1]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, men_means, width, label='Men')
        rects2 = ax.bar(x + width/2, women_means, width, label='Women')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Numbers')
        ax.set_title('Clustering gender distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')


        autolabel(rects1)
        autolabel(rects2)
        plt.show()
    
    def plot_cluster(self, arr_labels, centroids, savepath = None):
        colmap_clu = {0: 'r', 1: 'b', 2: 'g', 3:'y', 4:'DarkGreen', 5:'k', 6:'m', 7:'c', 8: 'LightBlue', 9:"Orange"}
        colors = list(map(lambda x: colmap_clu[x], arr_labels))   
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.xaxis.set_tick_params(labelsize=22)
        ax.yaxis.set_tick_params(labelsize=22)
        ax.set_xlabel('$x_1$', fontsize = 22)
        ax.set_ylabel('$x_2$', fontsize = 22)
        ax.scatter(self.point_mat[:, 0], self.point_mat[:, 1], color=colors, alpha=0.2, edgecolor='k')
        
        for i in range(self.n_clusters):
            ax.scatter(centroids[i, 0], centroids[i, 1], color=colmap_clu[i])
#         plt.title("Clustering Visulization: red vs blue")
        plt.tight_layout()
        plt.show()
        if savepath:
            fig.savefig(savepath, bbox_inches='tight')

    ## compute demographic compositions for k clusters
    def compute_demo(self, arr_labels):
        count_demo = np.zeros([self.n_clusters, self.n_demoGroups])
        for k in range(self.n_clusters):
            for j in range(self.n_demoGroups):
                count_demo[k, j] = len(np.where((self.gender == j) & (arr_labels == k))[0])
        return count_demo
    
    ## compute balance
    def compute_balance(self, count_demo):
        '''
        Input
            count_demo: demographic compositions for k clusters
        
        Output:
            arr_balance: balance of k clusters
            np.min(arr_balance): overall cluster balance
            bn_cluster: the cluster index of minimum balance cluster
            bn_low: the minority demographic group in cluster bn_cluster
            bn_high: the majority demographic group in cluster bn_cluster
        
        '''
        
        max_count = np.max(count_demo, axis = 1)
        min_count = np.min(count_demo, axis = 1)
        arr_balance = min_count/(max_count + 1e-16)
  
        bn_cluster = np.argmin(arr_balance)
        bn_low = np.argmin(count_demo[bn_cluster, :])
        bn_high = np.argmax(count_demo[bn_cluster, :])
        
        return arr_balance, np.min(arr_balance), bn_cluster, bn_low, bn_high
    
    ## determine a target well-balanced cluster
    def choose_target_cluster(self, bn_cluster, bn_low, bn_high, count_demo, centroids, method = 'global'):
        if method == 'global':
            ## choose the cluster with the highest ratio between group bn_low and bn_high
            eps = 1e-16
            biased_cluster = np.argmax(count_demo[:, bn_low]/(count_demo[:, bn_high] + eps))
        elif method == 'local':
            ## choose the clutser closet to bn_cluster
            c1 = centroids[bn_cluster]
            dist = np.ones(self.n_clusters)*float('inf')
            for i in range(self.n_clusters):
                if i == bn_cluster: continue
                dist[i] = np.linalg.norm(centroids[i] - c1)
            assert np.min(dist) != float('inf')
            biased_cluster = np.argmin(dist)
        return biased_cluster
    
    
    ## Three clustering functions
    '''
    curr_label + curr_centers => clustering cost directly
    '''
    def update_objective_f1(self, arr_label, centroids):
        sum_dist = 0
        for i in range(self.n_clusters):
            idx = np.where(arr_label == i)[0]
            data_ci = self.point_mat[idx, 0:self.dim_n]
            if len(data_ci) == 0:
#                 assert len(data_ci) == 0
                continue
            sum_dist += np.sum(np.linalg.norm(data_ci - centroids[i, :], axis = 1)**2)
#             sum_dist += np.sum(ecdist(data_ci, centroids[i, :].reshape(1, -1), squared =True))
        return sum_dist/self.n_samples

    '''
    current labels => update true centers by average => compute clustering cost according to current labels

    '''
    def update_objective_f1_trueCenter(self, arr_label):
        updatedCenter = np.zeros((self.n_clusters, self.n_samples))
        sum_dist = 0.0
        for i in range(self.n_clusters):
            idx_ci = np.where(arr_label == i)[0]
            if len(idx_ci) == 0:
                continue
            center = np.mean(self.point_mat[idx_ci, 0:self.dim_n], axis = 0, keepdims = True)
            sum_dist += np.sum(np.linalg.norm(self.point_mat[idx_ci, 0:self.dim_n] - center, axis = 1)**2)
        return sum_dist/self.n_samples

    '''
    curr_centers => re-assign every points to its closest center => average cost/distance
    '''
    def update_objective_f1_kmeans(self, centroids):
        arr_dist = self.distance(self.point_mat[:, 0:self.dim_n], centroids)
        sum_dist = np.mean(np.min(arr_dist, axis = 1))
        return sum_dist
    
    ''' Inexact update of centers for pure kmeans updates'''
    def update_centers_kmeans(self, batch_mat, arr_label, curr_centroids, count_demo, alpha):
        counter = np.sum(count_demo, axis = 1)
        bacth_dist = self.distance(batch_mat, curr_centroids)
        closest_center_idx = np.argmin(bacth_dist, axis = 1)
        for i in range(len(batch_mat)):
            idx = closest_center_idx[i]
            counter[idx] = counter[idx] + 1.0
            curr_centroids[idx, :] = curr_centroids[idx, :]*(1.0 - alpha/counter[idx]) + batch_mat[i, :]*alpha/counter[idx]
        return curr_centroids, closest_center_idx
    
    ''' Inexact update of centers for pure swap updates'''
    def update_centers_balance(self, arr_label, curr_centroids, in_idx, count_demo, alpha):
        '''
        Input
        arr_label：vector of clustering labels
        curr_centroids: the set of k centers 
        in_idx: index of in points for all clusters
        count_demo: current demographic compositions for k clusters before swap
        alpha: 1.0 by default, use to control the step size of center updates
        
        Output
        curr_centroids: the new set of k centers 
        '''
        counter = np.sum(count_demo, axis = 1)
        for i in range(self.n_clusters):
            if len(in_idx[i]) > 0:
                count_k = counter[i]*1.0
                for j in in_idx[i]:
                    curr_centroids[i, :] = curr_centroids[i, :]*(1.0 - alpha/count_k) + self.point_mat[j, 0:self.dim_n]*alpha/count_k
        return curr_centroids
    
    '''Exact update of centers'''
    def update_centers_kmeans_exact(self, kmeans_batch_idx, arr_label, curr_centroids):
        arr_label = arr_label.astype('int') 
        old_label = arr_label[kmeans_batch_idx]

        bacth_dist = self.distance(self.point_mat[kmeans_batch_idx, 0:self.dim_n], curr_centroids)
        new_label = np.argmin(bacth_dist, axis = 1) + 1

        old_count = collections.Counter(arr_label)
        arr_label[kmeans_batch_idx] = new_label
        new_count = collections.Counter(arr_label)

        in_idx = defaultdict(list)
        out_idx = defaultdict(list)
        
        for i in range(len(kmeans_batch_idx)):
            if old_label[i] != new_label[i]:
                in_idx[new_label[i]] += [kmeans_batch_idx[i]]
                out_idx[old_label[i]] += [kmeans_batch_idx[i]]

        for i in range(self.n_clusters):
            if len(out_idx[i]) > 0 and len(in_idx[i]) > 0:
                curr_centroids[i, :] = curr_centroids[i, :]*old_count[i]/new_count[i] + (np.sum(self.point_mat[in_idx[i], 0:self.dim_n], axis = 0) - np.sum(self.point_mat[out_idx[i], 0:self.dim_n], axis = 0))/new_count[i]
            elif len(out_idx[i]) > 0:
                curr_centroids[i, :] = (curr_centroids[i, :]*old_count[i] - np.sum(self.point_mat[out_idx[i], 0:self.dim_n], axis = 0))/new_count[i]
            elif len(in_idx[i]) > 0:
                curr_centroids[i, :] = (curr_centroids[i, :]*old_count[i] + np.sum(self.point_mat[in_idx[i], 0:self.dim_n], axis = 0))/new_count[i]

        return curr_centroids, arr_label

    def update_centers_balance_exact(self, old_label, arr_label, curr_centroids, in_idx, out_idx):
        '''
        =========== Input ================
        old_label: old assignment before swap (this is to update balance in-time and avoid jump around)
        out_idx: index of out points for all clusters
        in_idx: index of in points for all clusters
        curr_centroids: dictionary of the set of k centers 
        arr_label： current assignment label vector
        =========== Output ================
        centroids: new center dictionary
        '''
        old_count = collections.Counter(old_label)
        new_count = collections.Counter(arr_label)
        for i in range(self.n_clusters):
            if len(out_idx[i]) > 0 and len(in_idx[i]) > 0:
                curr_centroids[i, :] = curr_centroids[i, :]*old_count[i]/new_count[i] + (np.sum(self.point_mat[in_idx[i], 0:self.dim_n], axis = 0) - np.sum(self.point_mat[out_idx[i], 0:self.dim_n], axis = 0))/new_count[i]
            elif len(out_idx[i]) > 0:
                curr_centroids[i, :] = (curr_centroids[i, :]*old_count[i] - np.sum(self.point_mat[out_idx[i], 0:self.dim_n], axis = 0))/new_count[i]
            elif len(in_idx[i]) > 0:
                curr_centroids[i, :] = (curr_centroids[i, :]*old_count[i] + np.sum(self.point_mat[in_idx[i], 0:self.dim_n], axis = 0))/new_count[i]
        return curr_centroids
    
    ''' Batch update of centers for pure k-means updates'''
    def update_centers_kmeans_batch(self, batch_mat, arr_label, curr_centroids, count_demo, alpha):
        counter = np.sum(count_demo, axis = 1)
        bacth_dist = self.distance(batch_mat, curr_centroids)
        closest_center_idx = np.argmin(bacth_dist, axis = 1)

        for i in range(self.n_clusters):
            index = np.where(closest_center_idx == i)[0]
            if len(index) > 0:
                counter[i] += len(index)
                aver_direction = np.sum(curr_centroids[i].reshape(1, -1) - batch_mat[index,:], axis = 0)
                curr_centroids[i, :] = curr_centroids[i, :] - aver_direction/counter[i]*1.0
        return curr_centroids, closest_center_idx

    ''' Batch update of centers for pure swap updates'''
    def update_centers_balance_batch(self, arr_label, curr_centroids, in_idx, count_demo, alpha):
        '''
        Input
        arr_label：vector of clustering labels
        curr_centroids: the set of k centers 
        in_idx: index of in points for all clusters
        count_demo: current demographic compositions for k clusters before swap
        alpha: 1.0 by default, use to control the step size of center updates
        
        Output
        curr_centroids: the new set of k centers 
        '''

        counter = np.sum(count_demo, axis = 1)
        for i in range(self.n_clusters):
            if len(in_idx[i]) > 0:
                aver_direction = np.sum(curr_centroids[i].reshape(1, -1) - self.point_mat[in_idx[i], 0:self.dim_n], axis = 0)
                curr_centroids[i] = curr_centroids[i] - aver_direction/counter[i]*1.0

        return curr_centroids
    
    ## choose point(s) that is closest to the target cluster to swap
    def swap_index_selection(self, candidates_idx, num_batch, num_res, label, centroids):
        batch_idx = np.random.choice(candidates_idx, num_batch)
        batch_mat = self.point_mat[batch_idx, 0:self.dim_n]
        arr_dist = np.linalg.norm(batch_mat - centroids[label, :], axis = 1)
        return batch_idx[np.argmin(arr_dist)]
    
    
    def batch_fair_kmeans_descent(self, med = 'global', kmean_batch = 4, max_swap = 2, max_iters = 500):
        '''
        input:
            med: methods for choosing the target cluster. Available choice: "global"(by default), "local".
            kmean_batch: number of k-means updates in each outer iteration
            max_swap: number of swap updates in each outer iteration
            max_iters: total number of outer iterations
            
        output:
            kmean_f1: the sequence of k-means costs using updated centers
            kmean_f1_lables: the sequence of k-means cost using the exact centers computed by current labels
            balance_f2: the sequence of balances
            arr_curr_centroids: the sequence of centers
            arr_curr_label: the sequence of labels
        
        '''
        
        
        alpha = 1.0
        eps = 1e-15
        
        curr_label = self.arr_init_labels[0, :] + 0
        curr_centroids = self.arr_init_centroids[0, :, :]*1.0

        kmean_f1 = np.zeros(max_iters + 1)
        kmean_f1_lables = np.zeros(max_iters + 1)
        balance_f2 = np.zeros(max_iters + 1)
        arr_curr_centroids = np.zeros((max_iters + 1, self.n_clusters, self.dim_n))
        arr_curr_label = np.zeros((max_iters + 1, self.n_samples))
        
        arr_curr_centroids[0, :, :] = curr_centroids
        arr_curr_label[0, :] = curr_label
        
        
        self.count_clu = collections.Counter(curr_label)
        
        kmean_f1[0] = self.update_objective_f1(curr_label, curr_centroids)
        kmean_f1_lables[0] = self.update_objective_f1_trueCenter(curr_label)
        
        count_demo = self.compute_demo(curr_label)
        _, balance_f2[0], bn_cluster, bn_low, bn_high = self.compute_balance(count_demo)
        biased_cluster = self.choose_target_cluster(bn_cluster, bn_low, bn_high, count_demo, curr_centroids, method = med)

        kmean_epoch_idx = np.random.permutation(self.n_samples)
        kmean_step_count = 0

        swap_indicator_vector = np.zeros([self.n_clusters, self.n_samples])
        reset_count = 0
        
        
        iters = 1
        while iters <= max_iters:
            
            ## do max_swap steps of pure swap updates
            num_batch = min(self.n_samples, int(1.005**(iters // 2)))
            swap_success_count = 0
            in_idx = defaultdict(list)
            out_idx = defaultdict(list)
            old_label = curr_label + 0
            
            if max_swap > 0:
                ## check if the minimum balanced cluster changes or not
                if np.argmin(np.min(count_demo, axis = 1)/np.max(count_demo, axis = 1)) != bn_cluster:
                    _, balance, bn_cluster, bn_low, bn_high = self.compute_balance(count_demo)
                else:
                    bn_low = np.argmin(count_demo[bn_cluster, :])
                    bn_high = np.argmax(count_demo[bn_cluster, :])
                biased_cluster = self.choose_target_cluster(bn_cluster, bn_low, bn_high, count_demo, curr_centroids, method = med)
            
            while swap_success_count < max_swap:
                ## check if the target well-balanced cluster changes or not
                if swap_success_count > 0:
                    if med == 'global':
                        if np.argmax(count_demo[:, bn_low]/(count_demo[:, bn_high] + eps)) != biased_cluster:
                            biased_cluster = np.argmax(count_demo[:, bn_low]/(count_demo[:, bn_high] + eps))
                    elif med == 'local':
                        biased_cluster = self.choose_target_cluster(bn_cluster, bn_low, bn_high, count_demo, curr_centroids, method = med)
                
                candidates_idx2 = np.where((self.gender == bn_low) & (curr_label == biased_cluster) & (swap_indicator_vector[biased_cluster, :] == 0))[0] 
                if len(candidates_idx2) == 0:
                    swap_indicator_vector[biased_cluster, :] = np.zeros([1, self.n_samples])
                    reset_count += 1
                    candidates_idx2 = np.where((self.gender == bn_low) & (curr_label == biased_cluster))[0] 
                    
                idx_2 = self.swap_index_selection(candidates_idx2, min(num_batch, len(candidates_idx2)), 1, bn_cluster, curr_centroids) # np.random.choice(candidates_idx2, 1)
                in_idx[bn_cluster] += [idx_2] 
                out_idx[biased_cluster] += [idx_2]
                curr_label[idx_2] = bn_cluster

                count_demo[bn_cluster, bn_low] += 1
                count_demo[biased_cluster, bn_low] -= 1

                swap_success_count += 1
                swap_indicator_vector[bn_cluster, idx_2] = 1
                swap_indicator_vector[biased_cluster, idx_2] = 1
                
                if swap_success_count < max_swap:
                    ## check if the minimum balanced cluster changes or not
                    if np.argmin(np.min(count_demo, axis = 1)/np.max(count_demo, axis = 1)) != bn_cluster:
                        _, balance, bn_cluster, bn_low, bn_high = self.compute_balance(count_demo)
                    else:
                        bn_low = np.argmin(count_demo[bn_cluster, :])
                        bn_high = np.argmax(count_demo[bn_cluster, :])
                    biased_cluster = self.choose_target_cluster(bn_cluster, bn_low, bn_high, count_demo, curr_centroids, method = med)
                    
                    candidates_idx1 = np.where((self.gender == bn_high) & (curr_label == bn_cluster) & (swap_indicator_vector[bn_cluster, :] == 0))[0]
                    if len(candidates_idx1) == 0:
                        swap_indicator_vector[bn_cluster, :] = np.zeros([1, self.n_samples])
                        reset_count += 1
                        candidates_idx1 = np.where((self.gender == bn_high) & (curr_label == bn_cluster))[0]
                    
                    idx_1 = self.swap_index_selection(candidates_idx1, min(num_batch, len(candidates_idx1)), 1, biased_cluster, curr_centroids) #np.random.choice(candidates_idx1, 1)
                    in_idx[biased_cluster] += [idx_1]
                    out_idx[bn_cluster] += [idx_1]
                    curr_label[idx_1] = biased_cluster

                    count_demo[biased_cluster, bn_high] += 1
                    count_demo[bn_cluster, bn_high] -= 1

                    swap_success_count += 1
                    swap_indicator_vector[bn_cluster, idx_1] = 1
                    swap_indicator_vector[biased_cluster, idx_1] = 1

            if max_swap > 0:    
                curr_centroids = self.update_centers_balance_batch(curr_label, curr_centroids, in_idx, count_demo, alpha)
    
            ## do kmean_batch steps of pure kmeans updates
            if kmean_batch > 0:
                if kmean_batch + kmean_step_count > self.n_samples:
                    kmeans_batch_idx = kmean_epoch_idx[kmean_step_count:]
                    kmean_epoch_idx = np.random.permutation(self.n_samples) ## re-shuffle
                    kmean_step_count = 0
                else:
                    kmeans_batch_idx = kmean_epoch_idx[kmean_step_count:kmean_batch + kmean_step_count]
                    kmean_step_count += kmean_batch
                    
                point_mat_batch = self.point_mat[kmeans_batch_idx, :self.dim_n] 
                old_label_batch = curr_label[kmeans_batch_idx]
#                 curr_centroids, batch_center_idx = self.update_centers_kmeans(point_mat_batch, curr_label, curr_centroids, alpha)
                curr_centroids, batch_center_idx = self.update_centers_kmeans_batch(point_mat_batch, curr_label, curr_centroids, count_demo, alpha)
                curr_label[kmeans_batch_idx] = batch_center_idx
        
                for demo, old, new in zip(self.point_mat[kmeans_batch_idx, self.dim_n], old_label_batch, batch_center_idx):
                    if old != new:
                        count_demo[int(new), int(demo)] += 1
                        count_demo[int(old), int(demo)] -= 1 
    
            kmean_f1[iters] = self.update_objective_f1(curr_label, curr_centroids)
            _, balance_f2[iters], _ , _, _ = self.compute_balance(count_demo)

            kmean_f1_lables[iters] = self.update_objective_f1_trueCenter(curr_label)
            
            arr_curr_centroids[iters, :, :] = curr_centroids
            arr_curr_label[iters, :] = curr_label
            
            iters += 1
        print("Total number of outer iterations:", iters)
        return kmean_f1, kmean_f1_lables, balance_f2, arr_curr_centroids, arr_curr_label
    
    def stochastic_alt_descent(self, labels, centroids, count_demo, kmean_batch = 5, max_swap = 2, num_iters = 1, med = 'global'):
        '''
        do num_iters times alternative k-means and swap updates from the given clustering label and center.
        
        input:
            labels: given initial labels
            centroids: given initial centers
            count_demo: initial demographic compositions for k clusters
            kmean_batch: n_a, number of k-means updates per iteration
            max_swap: n_b, number of swap updates per iteration
            num_iters: number of iterations
            med: methods for choosing the target cluster. Available choice: "global"(by default), "local".
            
        output:
            kmean_f1: updated k-means cost using updated centers
            balance_f2: updated balance
            curr_centroid: updated centers
            curr_label: updated labels
            count_demo: updated demographic compositions
            kmean_f1_lables: updated k-means cost using the exact centers computed by current labels
            kmean_f1_centers: updated k-means cost using the exact clustering labels computed by current centers

        '''
        curr_label = labels+0
        curr_centroid = centroids+0
        eps = 1e-15
        
        _, balance, bn_cluster, bn_low, bn_high = self.compute_balance(count_demo)
        biased_cluster = self.choose_target_cluster(bn_cluster, bn_low, bn_high, count_demo, curr_centroid, method = med)
        num_batch = min(self.n_samples, int(1.003**(self.num_iter // 2)))
        
        for t in range(num_iters):
            
            ## do certain steps of pure swap updates
            swap_success_count = 0
            in_idx = defaultdict(list)
            out_idx = defaultdict(list)
            old_label = curr_label + 0
            
            if max_swap > 0:
                ## check if the minimum balanced cluster changes or not
                if np.argmin(np.min(count_demo, axis = 1)/np.max(count_demo, axis = 1)) != bn_cluster:
                    _, balance, bn_cluster, bn_low, bn_high = self.compute_balance(count_demo)
                else:
                    bn_low = np.argmin(count_demo[bn_cluster, :])
                    bn_high = np.argmax(count_demo[bn_cluster, :])
                biased_cluster = self.choose_target_cluster(bn_cluster, bn_low, bn_high, count_demo, curr_centroid, method = med)


            while swap_success_count < max_swap:
                
                ## check if the target cluster changes or not
                if swap_success_count > 0:
                    if med == 'global':
                        if np.argmax(count_demo[:, bn_low]/(count_demo[:, bn_high] + eps)) != biased_cluster:
                            biased_cluster = np.argmax(count_demo[:, bn_low]/(count_demo[:, bn_high] + eps))
                    elif med == 'local':
                        biased_cluster = self.choose_target_cluster(bn_cluster, bn_low, bn_high, count_demo, curr_centroid, method = med)
                
                candidates_idx2 = np.where((self.gender == bn_low) & (curr_label == biased_cluster))[0] 
                idx_2 = self.swap_index_selection(candidates_idx2, min(num_batch, len(candidates_idx2)), 1, bn_cluster, curr_centroid) # np.random.choice(candidates_idx2, 1)
                in_idx[bn_cluster] += [idx_2] 
                out_idx[biased_cluster] += [idx_2]
                curr_label[idx_2] = bn_cluster

                count_demo[bn_cluster, bn_low] += 1
                count_demo[biased_cluster, bn_low] -= 1

                swap_success_count += 1
                
                if swap_success_count < max_swap:
                    ## check if the minimum balanced cluster changes or not
                    if np.argmin(np.min(count_demo, axis = 1)/np.max(count_demo, axis = 1)) != bn_cluster:
                        _, balance, bn_cluster, bn_low, bn_high = self.compute_balance(count_demo)
                    else:
                        bn_low = np.argmin(count_demo[bn_cluster, :])
                        bn_high = np.argmax(count_demo[bn_cluster, :])
                    biased_cluster = self.choose_target_cluster(bn_cluster, bn_low, bn_high, count_demo, curr_centroid, method = med)
                    

                    candidates_idx1 = np.where((self.gender == bn_high) & (curr_label == bn_cluster))[0]
                    
                    idx_1 = self.swap_index_selection(candidates_idx1, min(num_batch, len(candidates_idx1)), 1, biased_cluster, curr_centroid) #np.random.choice(candidates_idx1, 1)
                    in_idx[biased_cluster] += [idx_1]
                    out_idx[bn_cluster] += [idx_1]
                    curr_label[idx_1] = biased_cluster

                    count_demo[biased_cluster, bn_high] += 1
                    count_demo[bn_cluster, bn_high] -= 1

                    swap_success_count += 1

            if max_swap > 0:
                curr_centroid = self.update_centers_balance_batch(curr_label, curr_centroid, in_idx, count_demo, self.stepsize)

             ## do certain steps of pure kmeans updates
            if kmean_batch > 0:
                kmeans_batch_idx = np.random.choice(self.n_samples, kmean_batch, replace=False)
                point_mat_batch = self.point_mat[kmeans_batch_idx, 0:self.dim_n] 
                old_label_batch = curr_label[kmeans_batch_idx]
                curr_centroid, batch_center_idx = self.update_centers_kmeans_batch(point_mat_batch, curr_label, curr_centroid, count_demo, self.stepsize)
                curr_label[kmeans_batch_idx] = batch_center_idx
                
                ## update demo composition for k clusters
                for demo, old, new in zip(self.point_mat[kmeans_batch_idx, self.dim_n], old_label_batch, batch_center_idx):
                    if old != new:
                        count_demo[int(new), int(demo)] += 1
                        count_demo[int(old), int(demo)] -= 1
        
        kmean_f1 = 0 #self.update_objective_f1(curr_label, curr_centroid)
        _, balance_f2,_, _, _ = self.compute_balance(count_demo)

        kmean_f1_lables = self.update_objective_f1_trueCenter(curr_label)
        kmean_f1_centers= 0 #self.update_objective_f1_kmeans(curr_centroid)

        return kmean_f1, balance_f2, curr_centroid, curr_label, count_demo, kmean_f1_lables, kmean_f1_centers
    
    # remove non-dominated points from current list from dense region
    def remove_dense(self, list_f1, list_f2, list_centers, list_labels, list_demo):
        num_total_pts = len(list_f1)
        if num_total_pts <= 50:
            return list_f1, list_f2, list_centers, list_labels, list_demo
        if self.dense_threshold == 0:
            dense_threshold = 1.0/(200 + self.num_iter/2.0) # 1.0/(100 + self.num_iter/2.0) good for bank
        else:
            dense_threshold = self.dense_threshold
    
        index_f1 = np.argsort(list_f1)
        index_f2 = np.argsort(list_f2)
        temp_list_f1 = np.sort(list_f1)
        temp_list_f2 = np.sort(list_f2)
        
        min_gap_f1 = (temp_list_f1[-1] - temp_list_f1[0])*dense_threshold + 1e-16
        max_gap_f2 = (temp_list_f2[-1] - temp_list_f2[0])*dense_threshold + 1e-16

        diff_list_f1 = np.diff(temp_list_f1)
        diff_list_f2 = np.diff(temp_list_f2)
        keep_index_f1 = [0, num_total_pts - 1]
        keep_index_f2 = [0, num_total_pts - 1]
        
        i = 1
        while i < num_total_pts - 2: # keep the first and last ones
            j = i
            curr_gap = diff_list_f1[j]
            while curr_gap <= min_gap_f1 and j < num_total_pts - 2:
                j += 1
                curr_gap += diff_list_f1[j]
            keep_index_f1.append((j - i)//2 + i)

            if i < j:
                i = j
            else:
                i += 1

        i = 1
        while i < num_total_pts - 2: # keep the first and last ones
            j = i
            curr_gap = diff_list_f2[j]
            while curr_gap <= max_gap_f2 and j < num_total_pts - 2:
                j += 1
                curr_gap += diff_list_f2[j]
            keep_index_f2.append((j - i)//2 + i)

            if i < j:
                i = j
            else:
                i += 1
        keep_index_f1 = np.unique(keep_index_f1)
        keep_index_f2 = np.unique(keep_index_f2)
        keep_index = np.union1d(index_f1[keep_index_f1], index_f2[keep_index_f2])
        
        return list_f1[keep_index], list_f2[keep_index], list_centers[keep_index, :, :],     list_labels[keep_index, :], list_demo[keep_index, :, :]
    
    # Remove dominated points at the end of each iteration
    def clear(self, list_f1, list_f2, list_centers, list_labels, list_demo):
        num_total_pts = len(list_f1)
        if num_total_pts <= 50:
            return list_f1, list_f2, list_centers, list_labels, list_demo
        
        array_f1 = list_f1
        array_f2 = -list_f2 
        
        x_bar = np.repeat(array_f1.reshape(-1,1), len(list_f1),axis = 1)
        y_bar = np.repeat(array_f2.reshape(-1,1), len(list_f2),axis = 1)
        
        x_check1 = (array_f1 <= x_bar)
        x_check2 = (array_f1 < x_bar)
        
        y_check1 = (array_f2 <= y_bar)
        y_check2 = (array_f2 < y_bar)
        
        all_check1 = (x_check1 & y_check2)
        all_check2 = (x_check2 & y_check1)
        
        sum1 = all_check1.sum(axis = 1)
        sum2 = all_check2.sum(axis = 1)
        
        rest_index = np.array([i for i in range(len(list_f1)) if (sum1[i] < 1 or sum2[i] < 1)])
        
        return array_f1[rest_index], -array_f2[rest_index], list_centers[rest_index, :, :],     list_labels[rest_index, :], list_demo[rest_index, :, :]
    
    def main_PFSAD(self, seed = 1):
        '''
        main function for Pareto front version of SAfairKM algorithms
        
        seed: random seed for generating initial labels/centers
        '''
        if self.init_provided: ## run the algorithm from an intermediate non-dominated point list
            f1_value_list, f2_value_list = self.f1_list, self.f2_list 
        else:
            self.num_iter = 0
            
            ## call init functions to initialize a list of labels and centers
            self.init_labels_centers(seed)
            f1_value_list = np.zeros(self.n_init)
            f2_value_list = np.zeros(self.n_init)
            demo_mat_list = np.zeros((self.n_init, self.n_clusters, self.n_demoGroups))

            for i in range(self.n_init):
    #             f1_value_list[i] = self.update_objective_f1(self.arr_init_labels[i, :], self.arr_init_centroids[i, :, :])
                f1_value_list[i] = self.update_objective_f1_trueCenter(self.arr_init_labels[i, :])
                demo_mat_list[i, :, :] = self.compute_demo(self.arr_init_labels[i, :])
                _, f2_value_list[i], _, _, _ = self.compute_balance(demo_mat_list[i, :, :])
        
        updating_label_list = self.arr_init_labels 
        updating_center_list = self.arr_init_centroids
        
        if self.data_name in ['Adult', 'Bank']:
            pairs = [(25, 0), (20, 4), (15, 3), (5, 2)]
            num_pairs = 4
        else:
            pairs = [(1, 1), (5, 1), (2, 0)]
            num_pairs = 3
        
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        begin = time.time()
        while len(updating_label_list) < self.max_len_pareto_front and self.num_iter < self.max_iter:
            tic = time.time()

            ## Allocate memory for shared + repeated data 
            shared_updating_label_list = np.tile(updating_label_list, (self.point_per_iteration, 1))
            shared_updating_center_list = np.tile(updating_center_list, (self.point_per_iteration, 1, 1))
            shared_demo_mat_list = np.tile(demo_mat_list, (self.point_per_iteration, 1, 1))

            pool = multiprocessing.Pool(processes=16)
            for i in range(num_pairs):
                zipped_resi = pool.starmap(partial(self.stochastic_alt_descent, kmean_batch = pairs[i][0], max_swap = pairs[i][1]), zip(shared_updating_label_list, shared_updating_center_list, shared_demo_mat_list))
                if i == 0:
                    zipped_res = zipped_resi
                    continue
                zipped_res += zipped_resi
            zipped_stacked_res = np.asarray(np.vstack(zipped_res))

#             zipped_res1 = pool.starmap(partial(self.stochastic_alt_descent, kmean_batch = 5, max_swap = 1), zip(shared_updating_label_list, shared_updating_center_list, shared_demo_mat_list))
#             zipped_res2 = pool.starmap(partial(self.stochastic_alt_descent, kmean_batch = 2, max_swap = 0), zip(shared_updating_label_list, shared_updating_center_list, shared_demo_mat_list))
#             zipped_res3 = pool.starmap(partial(self.stochastic_alt_descent, kmean_batch = 1, max_swap = 1), zip(shared_updating_label_list, shared_updating_center_list, shared_demo_mat_list))
#             zipped_stacked_res = np.asarray(np.vstack(zipped_res1 + zipped_res2 + zipped_res3))
            
            pool.close()
            pool.join()
            pool.terminate()
            
            f1_value_list = np.append(f1_value_list, zipped_stacked_res[:, 5])
            f2_value_list = np.append(f2_value_list, zipped_stacked_res[:, 1])
            updating_center_list = np.vstack([updating_center_list, np.stack(zipped_stacked_res[:, 2])])
            updating_label_list = np.vstack([updating_label_list, np.stack(zipped_stacked_res[:, 3])])
            demo_mat_list = np.vstack([demo_mat_list, np.stack(zipped_stacked_res[:, 4])])


            f1_value_list, f2_value_list, updating_center_list, updating_label_list, demo_mat_list = self.clear(f1_value_list, f2_value_list, updating_center_list, updating_label_list, demo_mat_list)
            f1_value_list, f2_value_list, updating_center_list, updating_label_list, demo_mat_list = self.remove_dense(f1_value_list, f2_value_list, updating_center_list, updating_label_list, demo_mat_list)
            
            print("time: ",  time.time() - tic)
            self.num_iter += self.num_steps_per_point
            
            print("#Pts: ", len(f1_value_list), " #Iter: ", self.num_iter)
            
        total_time = time.time() - begin
        print("Total time: ",  total_time)
        
        return f1_value_list, f2_value_list, updating_center_list, updating_label_list, total_time 

