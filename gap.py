from string import punctuation
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
    return P

def fitness(x, y):
    return 1/(1+F(x,y))

def fitness_all(n_ind):
    P = population(n_ind,1,3)
    listfitness = []
    for i in range(n_ind):
        listfitness.append(fitness(P[i,0],P[i,1]))
    return listfitness


def run():
    n_ind = int(input("Numero de individuos: "))
    print(population(n_ind,1,3))
    P = population(n_ind,1,3)
    print(fitness_all(n_ind))
if __name__ == "__main__":
    run()