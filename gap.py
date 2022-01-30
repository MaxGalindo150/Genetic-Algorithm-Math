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
        P[i,0] = random.uniform(lim_inf, lim_sup) #compotente x del vector i
        P[i,1] = random.uniform(lim_inf, lim_sup) #compotente y del vector i
    return P


def run():
    n_ind = int(input("Numero de individuos: "))
    print(population(n_ind,1,3))

if __name__ == "__main__":
    run()