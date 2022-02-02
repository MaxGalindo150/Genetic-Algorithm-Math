from re import M
import time
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
def fitness_all(P_matrix):
    listfitness = []
    for i in range(len(P_matrix)):
        listfitness.append(fitness(P_matrix[i,0],P_matrix[i,1]))
    return listfitness

#selection
def selection(listfitness, P_matrix):
    max_value = max(listfitness)
    index = np.argmax(listfitness)
    points_survivorsC = []
    points_survivors = []
    for i in range(len(listfitness)):
        p = listfitness[i]/max_value
        if p >= 0.3:
           points_survivorsC.append(str(int(10000*P_matrix[i,0])))
           points_survivorsC.append(str(int(10000*P_matrix[i,1])))
           points_survivors.append(P_matrix[i,0])
           points_survivors.append(P_matrix[i,1])
    return points_survivorsC, points_survivors, max_value, index #surviving vectors are placed on a list 


#crossover
def crossover(points_survivorC): #k1=0, -4, -8
    offspring = []               #k2=0, 1, 2
    k1 = 0                       #i=0, 1, 2 
    k2 = 0
    for i in range(int(len(points_survivorC)/2)):
        n = random.randint(0,4)
        if len(points_survivorC) == 2:
            k1 = -1
            offspring.append(float(points_survivorC[i+k2]))
            offspring.append(float(points_survivorC[2*(i+1) + k1]))
               
        else:    
            offspring.append(float(points_survivorC[i+k2][0:n]+points_survivorC[2*(i+1) + k1][n:len(points_survivorC[2*(i+1) + k1])]))
            offspring.append(float(points_survivorC[i+1+k2][0:n]+points_survivorC[2*(i+1)+1+k1][n:len(points_survivorC[2*(i+1)+1+k1])]))
            k1 = k1-4
            k2 = k2+1  
    if len(points_survivorC) == 2:
        pass
    else:
        offspring.pop()      
        offspring.pop()
    return offspring

#mutation
def mutation(offspring):
    offspringM = []
    for i in range(len(offspring)):
        x = random.uniform(-1, 1)
        mut1 = 10000*round(norm.pdf(x, 1, 1),4)
        offspringM.append(round((offspring[i] + mut1)/10000,4))
    return offspringM

#family integration 
def integration(points_survivors, offspringM):
    family = points_survivors + offspringM
    return family 

#family matrix
def mf(offspringM):
    H = np.zeros((int(len(offspringM)/2),2))
    k=0
    for j in range(int(len(offspringM)/2)):
        H[j,0]=offspringM[2*j]
        H[j,1]=offspringM[j+k+1]
        k=k+1
    return H

#best point
def best(MF, index):
    best_vector = np.array([MF[index,0],MF[index,1]])
    return best_vector



def run():
    n_ind = int(input("Numero de individuos: "))
    lim_inf = float(input("Limite inferior de x y y: "))
    lim_sup = float(input("Limite superior de x y y: "))
    
    start = time.time()
    P = population(n_ind, lim_inf, lim_sup)
    max_fitness = 0
    g = 0
    

    while abs(max_fitness-1) > 0.01:
        print(P)    

        g += 1 
        
        listfitness = fitness_all(P)
    
        points_survivorsC, points_survivors, max_fitness, indexfalse = selection(listfitness, P)
        
        offspring = crossover(points_survivorsC)
        
        offspringM = mutation(offspring)

        family = integration(points_survivors, offspringM)        
        
        P = mf(family)
        fit_son = fitness_all(P)
        
        x, y, z, index = selection(fit_son, P)
        best_vector = best(P, index)
        
        

    end = time.time()
    print()
    print(P)
    best_vector = best(P, index)
    print("Optimized point: ", best_vector)
    print()
    print("Optimized value: ", F(best_vector[0],best_vector[1]))
    print()
    print("Number of generations: ", g)
    print("Time: ", (end-start))


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

   #selection.2
#def selection2(n_ind): #papas
 #   s = selection1(n_ind)
  #  row = int(len(s)/2)
   # S = np.zeros((row,2))
    #k=0
    #for j in range(row):
     #   S[j,0]=s[2*j]
      #  S[j,1]=s[j+k+1]
       # k=k+1
    #return S

#crossover1
#def crossover1(n_ind):
 #   S = selection2(n_ind)
  #  LS = []
   # LSn = []
    #for i in range(len(S)):
     #   LS.append(str(int(10000*S[i,0])))
      #  LS.append(str(int(10000*S[i,1])))
       # LSn.append(S[i,0])
        #LSn.append(S[i,1])
    #return LS, LSn
     