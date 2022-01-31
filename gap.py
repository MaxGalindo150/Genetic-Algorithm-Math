from re import M, S
import re
from string import punctuation
from this import s
from tkinter import E
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
        P[i,0] = round(random.uniform(lim_inf, lim_sup),4) #componet x of vector i
        P[i,1] = round(random.uniform(lim_inf, lim_sup),4) #componet y of vector i
    return P #matrix of pupulation

#fitness from one individual
def fitness(x, y):
    return round(1/(1+F(x,y)),4) #if an individual's fitness approaches 1 is a good candidate  

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

#crossover1
def crossover1(n_ind):
    S = selection2(n_ind)
    St = []
    for i in range(len(S)):
        St.append(str(S[i,0]))
        St.append(str(S[i,1]))
    return St

#crossover2
def crossover2(n_ind):
    P = crossover1(n_ind)
    H = []
    k1 = 0 
    k2 = 0
    for i in range(int(len(P)/2)):
        n = random.randint(0,4)
        H.append(float(P[i+k2][0:n]+P[2*(i+1) + k1][n:len(P[1])]))
        H.append(float(P[i+1+k2][0:n]+P[2*(i+1)+1+k1][n:len(P[1])]))
        k1 = k1-4
        k2 = k2+1
    H.pop()    
    H.pop()
    return P, H


#convert to binary
#def convertbin(n_ind):
 #   S = selection2(n_ind)
  #  E = []
   # for i in range(len(S)):
    #    E.append(bin(int(10000*S[i,0])))
     #   E.append(bin(int(10000*S[i,1])))   
    #return E

#crossover 
#def crossover(n_ind):
 #   P = convertbin(n_ind)
  #  H = []
   # k1 = 0 
    #k2 = 0
    #for i in range(int(len(P)/2)):
     #   n = random.randint(3,4)
      #  H.append(P[i+k2][0:n]+P[2*(i+1) + k1][n:len(P[1])])
       # H.append(P[i+1+k2][0:n]+P[2*(i+1)+1+k1][n:len(P[1])])
        #k1 = k1-4
        #k2 = k2+1
    #F = P + H
    #return H, P, F   


def run():
    n_ind = int(input("Numero de individuos: "))
    P, H = crossover2(n_ind)
    print(P)
    print()
    print(H)   
if __name__ == "__main__":
    run()