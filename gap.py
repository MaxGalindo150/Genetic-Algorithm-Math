from string import punctuation
from this import s
import numpy as np
import math
import random

#function to optimize 
def F(x,y):
    return x**2 + y**2 

#create population  
def population(n_ind, lim_inf, lim_sup):
    P = np.zeros((n_ind,2))
    for i in range(n_ind):
        P[i,0] = random.uniform(lim_inf, lim_sup) #componet x of vector i
        P[i,1] = random.uniform(lim_inf, lim_sup) #componet y of vector i
    return P #matrix of pupulation

#fitness from one individual
def fitness(x, y):
    return 1/(1+F(x,y)) #if an individual's fitness approaches 1 is a good candidate  

#fitness from all individuals
def fitness_all(n_ind):
    P = population(n_ind,-6,6)
    listfitness = []
    for i in range(n_ind):
        listfitness.append(fitness(P[i,0],P[i,1]))
    return listfitness, P

#selection.1
def selection1(n_ind):
    L, P = fitness_all(n_ind)
    max_value = max(L)
    s = []
    for i in range(len(L)):
        p = L[i]/max_value
        if p >= 0.3:
           s.append(P[i,0])
           s.append(P[i,1])
    return s

#selection.2
def selection2(n_ind):
    s = selection1(n_ind)
    row = int(len(s)/2)
    S = np.zeros((row,2))
    k=0
    for j in range(row):
        S[j,0]=s[2*j]
        S[j,1]=s[j+k+1]
        k=k+1
    return S

def run():
    n_ind = int(input("Numero de individuos: "))
    S = selection2(n_ind)

    print(S)
    


if __name__ == "__main__":
    run()