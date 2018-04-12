# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 18:39:08 2018

@author: Shubham
"""

# Python code for FINAL 

# TODO-
# 1. Include sparse matrix functions for large adjacency lists
# 2. Input filename is in the form of JSON
#3. Line 67 to be incorporated/checked


import numpy as np
import numpy.linalg as LA 
import sys
import pandas as pd
from sklearn.preprocessing import *
from collections import defaultdict
import json
import pickle
from glob import glob
import networkx as nx
from graphcombinator import *
import os
import matplotlib.pyplot as plt

# global path to all the jsons
cpath = 'C:\\Users\\Shubham\\Desktop\\video-context-transcription\\src\\python\\video3_out'
        
# loads all the jsons
def load_json(cpath):
    json_file_list = [ele for ele in os.listdir(cpath) if ele.endswith('.json')] 
    rename_file = map(lambda x: cpath + '\\' + x, json_file_list)
    json_list = map(lambda x: json.load(open(x, 'r')), rename_file)
    return list(json_list)

# return graphs for all the jsons
def load_graphs(cpath):
    graphs = combine_scene_graphs_list(cpath)
    return graphs

# class defining FINAL algorithm
class FINAL():
    
    def __init__(self,alpha,maxiter,tol,graphs,cpath,json):
        self.alpha = alpha
        self.maxiter = maxiter
        self.tol = tol
        self.graphs = graphs
        self.cpath = cpath
        self.json = json
    
    # the two matrices here will load a numpy matrix for each matrix_processing and edge_attr_processing
    def matrix_processing(self,i,j):
        A1 = nx.adjacency_matrix(self.graphs[i], nodelist = list(self.graphs[i].nodes()))
        node1 = list(self.graphs[i].nodes())
        A2 = nx.adjacency_matrix(self.graphs[j], nodelist = list(self.graphs[j].nodes()))
        node2 = list(self.graphs[j].nodes())
        return A1, A2, node1, node2
    
    def edge_attr_processing(self,i,j):
        N1 = np.loadtxt(self.json[i], dtype = 'float')
        N2 = np.loadtxt(self.json[j], dtype = 'float')
        return N1,N2
    
    def node_attr_processing(self, filename):
        J1 = np.loadtxt(self.filename1, dtype = 'float')
        J2 = np.loadtxt(self.filename2, dtype = 'float')
        
         # instantiation of the OneHotEncoded form
        enc = LabelEncoder()
        
        # --------------- The one-hot encoding process -----------------------
        N1_enc = enc.fit(N1)
        N2_enc = enc.fit(N2)
        
        N1_enc = N1_enc.reshape(-1,1)
        N2_enc = N2_enc.reshape(-1,1)
        
        # --------------------------------------------------------------------
        
        # instantiation of the OneHotEncoded form
        ohe = OneHotEncoder(sparse = False)
        
        # N1 and N2 are finally n1 X k and n2 X k matrices respectively
        N1_enc = ohe.fit_tranform(N1_enc)
        N2_enc = ohe.fit_tranform(N2_enc)
        return N1_enc, N2_enc
    
#     def relationship(filename):

#         objects = defaultdict(str)
#         relationship = defaultdict(str)

#         # lst will different Jon objects
#         for i in range(len(lst)):
#             objects[i] = lst[i]['objects']
#             relationship[i] = (lst[i]['relationships'][i]['predicate'])

        # print objects, relationship 

    
    def Final(self,A1, A2, N1, N2, E1, E2, H):
        n1 = A1.shape[0]
        n2 = A2.shape[0]
       
        # Initializing N1 and N2 if they are empty
        if N1.shape[0] + N2.shape[0] == 0:
            N1 = np.ones(n1, 1)
            N2 = np.ones(n2, 1)
        
        # Initializing E1 and E2 if they are empty
        if E1.shape[0] + E2.shape[0] == 0:
            E1 = np.zeros([1,n1,n1])
            E2 = np.zeros([1,n2,n2])
            E1[1,:,:] = A1
            E2[1,:,:] = A2
        
        
        # the number of unique edge attributes
        K = N1.shape[2]
        L = E1.shape[2]
        
        
        # Normalize edge feature vectors
        T1 = np.zeros([n1,n1])
        T2 = np.zeros([n2,n2])
        
        for l in range(L):
            T1 = T1 + E1[l,:,:]**2
            T2 = T2 + E2[l,:,:]**2
            
        T1_map = map(lambda x: np.sqrt(x), T1)
        T1 = np.fromiter(T1_map, dtype = float)
        
        T2_map = map(lambda x: np.sqrt(x), T2)
        T2 = np.fromiter(T2_map, dtype = float)
        
        for l in range(L):
            E1[l,:,:] = np.mutliply(E1[l:,:,],T1)
            E2[l,:,:] = np.mutliply(E2[l:,:,],T2)
    
    
        # normalising node features vectors
            # taking sum about second axis
        K1 = np.sum(N1**2,1)**(-0.5)
        K2 = np.sum(N2**2,1) **(-0.5)
        
            # matrix multiplication check the order
        for i in range(n1):
            N1[i,:] = np.multiply(N1[i,:],K1)
            N2[i,:] = np.multiply(N2[i,:],K2)
        
        
        # calculating kronecker product for computing node feature cosine cross similarity
        N = np.zeros([n1*n2,1])
        for k in range(K):
            N = N + np.kron(N1[:,k], N2[:,k])
        
        
        # compute Kronecker Degree vector
        d = np.zeros([n1*n2,1])
        for l in range(L):
            for k in range(K):
                d = d + np.kron(np.multiply(E1[1,:,:], A1) * N1[:,k],  np.multiply(E2[1,:,:], A2)* N2[:,k])
                
    
        D = np.multiply(N, d)
        DD = D**(-0.5)
        DD = np.where(D == 0, 0, DD)
        
        #fixed point solution
        q = np.multiply(DD, N)
        
        # assumption is that H is randomly intialized, this can be changed later on
        H_vec = np.vectorize(H)
        s = H_vec
        
        # optimization to obtain similarity matrix
        for i in range(self.maxIter):
            prev = s
            M = np.reshape(np.multiply(q,s), [n2,n1])
            # the algorithm implied sparse but blah
            S =  np.zeros([n2,n1])
            for l in range(L):
                S +=  np.multiply(E2[:,l], A2) * M * np.multiply(E1[:,l], A1)
            
            s = (1 - self.alpha) * h * self.alpha * np.multiply(q, np.vectorize(S))
            diff = np.norm(s - prev)
            
            if diff < self.tol:
                break
                
        # S is the similarity matrix
        S = np.reshape(s,[n2,n1])
        return S
    
    # Graph Similarity measure for now which is the first two singular values or else check also the second last value
    def Graph_Sim(self, S):
        
        [U,D,V] = LA.svd(S)
        a = float(D[1])/float(D[0])
        return a
    

d = load_graphs(cpath)
s = len(d[0].nodes())
t = len(d[1].nodes())
s1 = (d[0].nodes())
t1 = (d[1].nodes())
e = set(s1).union(set(t1))
#print(len(t1))
#print(len(list(e)))
        
#color = nx.get_edge_attributes(d[0])
