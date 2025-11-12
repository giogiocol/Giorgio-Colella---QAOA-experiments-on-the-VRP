import pandas as pd
import numpy as np
import tabulate as tab
import random
import networkx as nx
import matplotlib.pyplot as plt
import csv
import math

def build_matrix_A_adj(N, X_ij_to_k):

    #CONSTRAINT 0
    A_C0 = np.empty((0, N*(N-1)))

    row = np.zeros(N*(N-1))
    for j in range(N):
        if j != 0:
            row[X_ij_to_k[(0, j)]] = 1

    A_C0 = np.vstack((A_C0, row))

    row = np.zeros(N*(N-1))
    for i in range(N):
        if i != 0:
            row[X_ij_to_k[(i, 0)]] = 1

    A_C0 = np.vstack((A_C0, row))

    #CONSTRAINT 1
    A_C1 = np.empty((0, N*(N-1)))

    for i in range(1, N):
        row = np.zeros(N*(N-1))

        for j in range(N):
            if i != j:
                row[X_ij_to_k[(i, j)]] = 1

        A_C1 = np.vstack((A_C1, row))

    #CONSTRAINT 2
    A_C2 = np.empty((0, N*(N-1)))

    for j in range(1, N):
        row = np.zeros(N*(N-1))

        for i in range(N):
            if i != j:
                row[X_ij_to_k[(i, j)]] = 1

        A_C2 = np.vstack((A_C2, row))

    #Matrix composition
    A = np.vstack((A_C0, A_C1, A_C2))

    return A, A_C0, A_C1, A_C2


def build_vect_C_adj(N, X_ij_to_k, d_ij, pen): #penalty is the coefficient for the penalty component of the obj. function

    C = np.zeros((N*(N-1), N*(N-1)))

    for i in range(1, N):
        for j in range(1, N):
            if i != j:
                C[X_ij_to_k[(i, j)], X_ij_to_k[i, j]] = d_ij[(i, j)]

                C[X_ij_to_k[(i, j)], X_ij_to_k[j, i]] = pen*max(d_ij.values()) 


    return C


def build_matrix_P_adj(A0, A1, A2, d_ij, alpha): 

    p = alpha*max(d_ij.values())
    P0 = p*np.ones(A0.shape[0])
    P1 = p*np.ones(A1.shape[0])
    P2 = p*np.ones(A2.shape[0])
    P = np.concatenate((P0, P1, P2), axis=0)

    P = np.diag(P)

    return P