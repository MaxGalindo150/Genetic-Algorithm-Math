from string import punctuation
import numpy as np
import math
import random
from scipy.stats import norm 

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
def selection2(n_ind): #papas
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
    LS = []
    LSn = []
    for i in range(len(S)):
        LS.append(str(int(10000*S[i,0])))
        LS.append(str(int(10000*S[i,1])))
        LSn.append(S[i,0])
        LSn.append(S[i,1])
    return LS, LSn

#crossover2
def crossover2(n_ind):
    LS, LSn = crossover1(n_ind)
    H = []
    k1 = 0 
    k2 = 0
    for i in range(int(len(LS)/2)):
        n = random.randint(0,4)
        if len(LS) == 2:
            k1 = -1
            H.append(float(LS[i+k2][0:n]+LS[2*(i+1) + k1][n:len(LS[2*(i+1) + k1])]))
        else:    
            H.append(float(LS[i+k2][0:n]+LS[2*(i+1) + k1][n:len(LS[2*(i+1) + k1])]))
            H.append(float(LS[i+1+k2][0:n]+LS[2*(i+1)+1+k1][n:len(LS[2*(i+1)+1+k1])]))
            k1 = k1-4
            k2 = k2+1  
    if len(LS) == 2:
        pass
    else:
        H.pop()      
        H.pop()
    return H

#mutation
def mutation(n_ind):
    H = crossover2(n_ind)
    MT = []
    for i in range(len(H)):
        x = random.uniform(-1, 1)
        mut1 = 10000*round(norm.pdf(x, 1, 1),4)
        MT.append(round((H[i] + mut1)/10000,4))
    return MT

#family integration 
def integration(n_ind):
    LS, LSn = crossover1(n_ind)
    MT = mutation(n_ind)
    return LSn + MT

#family matrix
def mf(n_ind):
    F = integration(n_ind)
    MF = np.zeros((int(len(F)/2),2))
    k=0
    for j in range(int(len(F)/2)):
        MF[j,0]=F[2*j]
        MF[j,1]=F[j+k+1]
        k=k+1
    return MF



def run():
    n_ind = int(input("Numero de individuos: "))


    


if __name__ == "__main__":
    run()



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
   
   #k=0
    #if len(H) == 1:
     #   x1 = random.uniform(-1, 1)
      #  mut = 10000*round(norm.pdf(x1, 1, 1),4)
       # H = (H + mut)/10000
        #MT = H
    #else:    
     #   for i in range(int(len(H)/2)):
      #      x = random.uniform(-1, 1)
       #     mut1 = 10000*round(norm.pdf(x, 1, 1),4)
        #    mut2 = 10000*round(norm.pdf(x, 1, 1),4)
         #   M[i,0]=round((H[2*i] + mut1)/10000,4)
          #  M[i,1]=round((H[i+k+1] + mut2)/10000,4)
           # k=k+1
        #MT = M    