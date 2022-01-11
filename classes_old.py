# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 14:51:09 2021

@author: njeannin
"""

import numpy as np 

class mat:
    c=100#display coefficient
    EI_min = 52
    EI_max = 430
    C_critic = 0.015
    
    # Instance method
    def init_EI(self):
        """ If the material behavior is gien though a table, this initializse 
        the value of EI depending of the first row of the table """
        
        i=0
        while self.MK[i,0]<0:
            i+=1 #find the point (0,0)
        EI = (self.MK[i+1,1]-self.MK[i,1])/(self.MK[i+1,0]-self.MK[i,0])
        
        return EI
    
    def __init__(self, EA=30463903.24, EI=52, MK=[] ):
        self.EA = EA #Tensile stiffness
        self.MK = np.array(MK) #Moment curvature variation table
        self.EI = EI #Bending stiffness
        if len(self.MK)==0:
            self.EI = EI #Bending stiffness
        else:    
            self.EI = self.int_EI()  

    


class mesh:
    
    def __init__(self,L,discr):
        self.L = L
        self.discr = discr
        self.nodes = np.array([np.linspace(0, self.L, self.discr+1),
                               np.zeros((self.discr+1))]).transpose()
        self.elements =  np.array([[i for i in range(self.discr)], 
                                   [i for i in range(1,self.discr+1)]])
        self.nb_element = self.elements.shape[0]

class field:
    
    def __init__(self, imp):
        self.imp = imp #prescribed
        #self.comp = np.zeros((3*self.mesh.nodes.shape[0]))
        #self.results = np.zeros((3*self.mesh.nodes.shape[0])) # result



class s:        
    
    def __init__(self, mat, mesh, U, F):
        self.mat = mat
        self.mesh = mesh
        self.EI = np.array([self.mat.EI]*self.mesh.nb_element)
        self.U = U
        self.F = F
        self.KL = np.zeros((3*self.mesh.nodes.shape[0],
                            3*self.mesh.nodes.shape[0]))
        self.K22 = np.zeros((len(self.U.imp),len(self.U.imp)))
        self.K11 = np.zeros((3*self.mesh.nodes.shape[0]-len(self.U.imp),
                             3*self.mesh.nodes.shape[0]-len(self.U.imp)))
        self.K12 = np.zeros((len(self.U.imp),
                             3*self.mesh.nodes.shape[0]-len(self.U.imp)))
        self.K21 = np.zeros((3*self.mesh.nodes.shape[0]-len(self.U.imp),
                             len(self.U.imp)))
        self.convergence=0
        self.U.results = np.zeros((3*self.mesh.nodes.shape[0]))
        self.F.results = np.zeros((3*self.mesh.nodes.shape[0]))
        
    def compute_free_sys(self):
        for e in range(self.mesh.nb_element):
            element_nodes = self.mesh.elements[e]
            #index of the nodes compoing the element
            nodes_coords = self.mesh.nodes[element_nodes] 
            #coordinates of thenodes composing the element
            L_e = np.sqrt((nodes_coords[1,0]-nodes_coords[0,0])**2
                          +(nodes_coords[1,1]-nodes_coords[0,1])**2)
            #print(L_e)
            EI_e = self.EI[e]
            Ku_e = (self.mat.EA/L_e)*np.array([[1,-1],[-1,1]])
            Kv_e = (EI_e/L_e**3)*np.array([[12, 6*L_e,-12,6*L_e],
                                           [6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
                                           [-12, -6*L_e,12,-6*L_e],
                                           [6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]])
            #transfer matrix to put the dof in the right order                    
            P=np.array([[1,0,0,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,0,1,0,0],
                        [0,1,0,0,0,0],
                        [0,0,0,0,1,0],
                        [0,0,0,0,0,1]])
            #Stifness matrix 
            K_e = np.zeros((6,6))
            K_e[:2,:2]=Ku_e
            K_e[2:,2:]=Kv_e
            K_e = np.dot(np.dot(P,K_e), np.linalg.inv(P))
            
            #transfert matrix to the global coordinate system
            lbd = (nodes_coords[1,0]-nodes_coords[0,0])/L_e
            mu = (nodes_coords[1,1]-nodes_coords[0,1])/L_e
            T = np.array([[lbd,mu,0,0,0,0],
                          [-mu,lbd,0,0,0,0],
                          [0,0,1,0,0,0],
                          [0,0,0,lbd,mu,0],
                          [0,0,0,-mu,lbd,0],
                          [0,0,0,0,0,1]])
            
            K_e = np.dot(np.dot(T,K_e), np.linalg.inv(T))
            
            #Assembly
            i,j = 3*element_nodes[0],3*element_nodes[1]
            self.KL[i:i+3,i:i+3]= K_e[:3,:3]
            self.KL[i:i+3,j:j+3]= K_e[:3,-3:]
            self.KL[j:j+3,i:i+3]= K_e[-3:,:3]
            self.KL[j:j+3,j:j+3]= K_e[-3:,-3:]
            #print (self.KL)
        return None
    
    def re_order_K(self):
        
        self.idU_imp = 3*self.U.imp[:,0] + self.U.imp[:,1]
        mask = np.ones(3*len(self.mesh.nodes), bool)
        mask[self.idU_imp]= False             
        self.idU_com = np.array([i for i in range(3*len(self.mesh.nodes))])[mask]
        
        self.K22 = self.KL[self.idU_imp,:][:,self.idU_imp]
        self.K11 = self.KL[self.idU_com,:][:,self.idU_com]
        self.K12 = self.KL[self.idU_com,:][:,self.idU_imp]
        self.K21 = self.KL[self.idU_imp,:][:,self.idU_com]
        
        return None
    
    def compute_results (self):
        F1 = np.zeros(3*len(self.mesh.nodes))
        for f in self.F.imp : 
            F1[3*int(f[0])+int(f[1])]=f[2]
        F1=F1[self.idU_com]
        #U2, F1 given blocs of the matrix
        #U1, U2 computed blocs 
        U1 = np.dot(np.linalg.inv(self.K11),(F1 - np.dot(self.K12, self.U.imp[:,2])))
        F2 = np.dot(self.K21, U1) + np.dot(self.K22, self.U.imp[:,2])
        
        self.U.results = np.zeros((3*self.mesh.nodes.shape[0]))
        self.F.results = np.zeros((3*self.mesh.nodes.shape[0]))
        self.U.results[self.idU_imp]= self.U.imp[:,2]
        self.U.results[self.idU_com]= U1
        self.F.results[self.idU_com]= F1
        self.F.results[self.idU_imp]= F2
        print(U1)
        return None
    
    def update_EI(self):
        C = np.zeros(self.mesh.nb_element)
        conv = True
        for e in range(self.mesh.nb_element):
            element_nodes = self.mesh.elements[e]
            #index of the nodes compoing the element
            nodes_coords = self.mesh.nodes[element_nodes] 
            #coordinates of thenodes composing the element
            L_e = np.sqrt((nodes_coords[1,0]-nodes_coords[0,0])**2
                          +(nodes_coords[1,1]-nodes_coords[0,1]**2))
            EI_e = self.EI[e].copy()
            #résultats en courbure 
            theta1 = self.U.results[3*element_nodes[0]+2]
            theta2 = self.U.results[3*element_nodes[1]+2]
            
            C[e]=(theta1-theta2)/L_e
            
            if C[e]<self.mat.C_critic:
                self.EI[e]=self.mat.EI_max
            else:
                self.EI[e]=self.mat.EI_min
            
            if (self.EI[e]-EI_e)>0.0001:
                conv = False #convergence check
        
        self.convergence = conv
        return None
    
    def get_results(self):
        #in progress 
        Ux = self.U.results[[i for i in range (0,len(self.U.results), 3)]]
        Uy = self.U.results[[i for i in range (1,len(self.U.results), 3)]]
        Utheta = self.U.results[[i for i in range (2,len(self.U.results), 3)]]
        Ux = Ux + self.mesh.nodes[:,0]
        Uy = Uy + self.mesh.nodes[:,1]
        return Ux, Uy, Utheta
    
    
        
                             
                            
            
        
        






