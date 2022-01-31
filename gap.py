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
    points_survivorsC = []
    points_survivors = []
    for i in range(len(listfitness)):
        p = listfitness[i]/max_value
        if p >= 0.3:
           points_survivorsC.append(str(int(10000*P_matrix[i,0])))
           points_survivorsC.append(str(int(10000*P_matrix[i,1])))
           points_survivors.append(P_matrix[i,0])
           points_survivors.append(P_matrix[i,1])
    return points_survivorsC, points_survivors #surviving vectors are placed on a list 


#crossover
def crossover(points_survivorC):
    offspring = []
    k1 = 0 
    k2 = 0
    for i in range(int(len(points_survivorC)/2)):
        n = random.randint(0,4)
        if len(points_survivorC) == 2:
            k1 = -1
            offspring.append(float(points_survivorC[i+k2][0:n]+points_survivorC[2*(i+1) + k1][n:len(points_survivorC[2*(i+1) + k1])]))
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
def mf(family):
    MF = np.zeros((int(len(family)/2),2))
    k=0
    for j in range(int(len(family)/2)):
        MF[j,0]=family[2*j]
        MF[j,1]=family[j+k+1]
        k=k+1
    return MF



def run():
    n_ind = int(input("Numero de individuos: "))
    lim_inf = float(input("Limite inferior de x y y: "))
    lim_sup = float(input("Limite superior de x y y: "))
    P = population(n_ind, lim_inf, lim_sup)
    print("Matriz inicial:")
    print(P)
    print()
    print("Fitness de los vectores padre:")
    listfitness = fitness_all(P)
    print(listfitness)
    print()
    print("Mejores fitness:")
    points_survivorsC, points_survivors = selection(listfitness, P)
    print()
    print(" Vectores seleccionados:")
    print()
    print(points_survivors)
    print()
    print(" Vectores seleccionados y preparados para cruzar:")
    print(points_survivorsC)
    print()
    print("cruce:")
    offspring = crossover(points_survivorsC)
    print(offspring)
    print()
    print("Mutacion:")
    offspringM = mutation(offspring)
    print(offspringM)
    print()
    print("juntamos padres e hijos:")
    family = integration(points_survivors, offspringM)
    print(family)
    print()
    print("hacemos la matriz de la familia:")
    MF = mf(family)
    print(MF)
    


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
     