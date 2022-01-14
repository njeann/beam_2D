# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 14:50:15 2021

@author: njeannin
"""

import numpy as np
import matplotlib.pyplot as plt
from classes import mat, mesh, field, s

# Initialization
L= 1
discr = 6


material = mat()
the_mesh = mesh(L,discr)

BCU = np.array([[0,0,0],[0,1,0],[discr,0,0],[discr,1,0]])
BCF = np.array([[0,2,10.0],[discr,2,-10.0]])

U = field(BCU.copy())
F = field(BCF.copy())
system = s(material, the_mesh, U, F)
nb_inc = 3

Ux, Uy, Utheta = system.get_results()
plt.plot(Ux,Uy, label= "initial")

# Calculation
U.imp[:,2] = 0
F.imp[:,2] = 0

for j in range(1,nb_inc+1): 
    U.imp[:,2] = U.imp[:,2] + BCU[:,2]/nb_inc
    F.imp[:,2] = F.imp[:,2] + BCF[:,2]/nb_inc
    #print(F.imp)
    system.convergence=False
    i=0 #increment
    while i<1000 and system.convergence==False : 
        system.assemble()
        system.re_order_K()
        
        #FE2
        Eps, Ki = system.compute_results()
        N,M = system.run_abaqus(Eps, Ki)
        system.update_EI(Eps, Ki, N, M)
        """
        #Simple
        system.compute_results()
        system.update_EI()
        """
        i+=1
    
    print(i,system.convergence)
    print(system.EI)
    Ux, Uy, Utheta = system.get_results()
    plt.plot(Ux,Uy, label= "inc"+str(j))
    system.M0 = M
    system.Ki0 = Ki

plt.legend()
plt.show()
