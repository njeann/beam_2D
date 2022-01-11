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
        self.L = L #length of the beam 
        self.discr = discr # number of discretisations
        self.nodes = np.array([np.linspace(0, self.L, self.discr+1),
                               np.zeros((self.discr+1))]).transpose() #array of nodes
        self.elements =  np.array([[i for i in range(self.discr)], 
                                   [i for i in range(1,self.discr+1)]]).transpose() #connections
        self.nb_element = self.elements.shape[0]

class field:
    
    def __init__(self, imp):
        self.imp = imp #prescribed
        #self.comp = np.zeros((3*self.mesh.nodes.shape[0])) #computed 
        #self.results = np.zeros((3*self.mesh.nodes.shape[0])) # result



class s:        
    
    def __init__(self, mat, mesh, U, F):
        self.mat = mat #materil
        self.mesh = mesh #mesh
        self.EI = np.array([self.mat.EI]*self.mesh.nb_element) #bending stifness for each element
        self.U = U #displacements
        self.F = F #forces
        self.KL = np.zeros((3*self.mesh.nodes.shape[0],
                            3*self.mesh.nodes.shape[0])) #assembled stifness matrix
        #sub matrixes
        self.K22 = np.zeros((len(self.U.imp),len(self.U.imp))) #known displacements
        self.K11 = np.zeros((3*self.mesh.nodes.shape[0]-len(self.U.imp),
                             3*self.mesh.nodes.shape[0]-len(self.U.imp)))#unknown displacement
        self.K12 = np.zeros((len(self.U.imp),
                             3*self.mesh.nodes.shape[0]-len(self.U.imp)))
        self.K21 = np.zeros((3*self.mesh.nodes.shape[0]-len(self.U.imp),
                             len(self.U.imp)))
        self.convergence=0 #convergence test 
        self.U.results = np.zeros((3*self.mesh.nodes.shape[0])) 
        self.F.results = np.zeros((3*self.mesh.nodes.shape[0]))
        self.M0 = np.zeros((self.mesh.nb_element)) #initialisation of elemental bending moment history 
        self.Ki0 = np.zeros((self.mesh.nb_element))#initialisation of elemental curvature history
        
    def compute_free_sys(self):
        """Assemble the elemental matrixes into the global stiffness matrix"""
        for e in range(self.mesh.nb_element):
            element_nodes = self.mesh.elements[e]
            #index of the nodes composing the element
            nodes_coords = self.mesh.nodes[element_nodes] 
            #coordinates of thenodes composing the element
            L_e = np.sqrt((nodes_coords[1,0]-nodes_coords[0,0])**2
                          +(nodes_coords[1,1]-nodes_coords[0,1])**2)#length of the element
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
        """Re order the stiffness matrix into 4 submatrixes 
        (sliced between known and unknown displacements)"""
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
        """solve the problem using the ordered submatrixes"""
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
        print(self.U.results)
        
        
        Ki = np.zeros((self.mesh.nb_element))
        Eps = np.zeros((self.mesh.nb_element))
        L_0 = self.mesh.L/ self.mesh.discr
        for i in range(self.mesh.nb_element):
            element_nodes = self.mesh.elements[i]
            #index of the nodes compoing the element
            nodes_coords = self.mesh.nodes[element_nodes] 
            #coordinates of thenodes composing the element
            L_e = np.sqrt((nodes_coords[1,0]-nodes_coords[0,0])**2
                          +(nodes_coords[1,1]-nodes_coords[0,1])**2)
            #résultats en courbure 
            theta1 = self.U.results[3*element_nodes[0]+2]
            theta2 = self.U.results[3*element_nodes[1]+2]
            
            Ki[i] = (theta1-theta2)/L_e
            Eps[i] = L_e/L_0
            
            
        return Eps,Ki
    
    def update_EI(self, Eps, Ki, N, M):
        """ubdate EI dependinf on the curvature and moment variation"""
        #self.Ki[:,1] = Ki
        #self.M[:,1] = M
        conv = True
        for e in range(self.mesh.nb_element):
            EI_e = self.EI[e].copy()
             #Si on part de zero
            self.EI[e] = abs((M[e]-self.M0[e])/(Ki[e]-self.Ki0[e]))
            
            if (self.EI[e]-EI_e)>0.1:
                conv = False #convergence check
        
        self.convergence = conv
        return None
    
    def get_results(self):
        """Conpute nodal results"""
        #in progress 
        Ux = self.U.results[[i for i in range (0,len(self.U.results), 3)]]
        Uy = self.U.results[[i for i in range (1,len(self.U.results), 3)]]
        Utheta = self.U.results[[i for i in range (2,len(self.U.results), 3)]]
        Ux = Ux + self.mesh.nodes[:,0]
        Uy = Uy + self.mesh.nodes[:,1]
        return Ux, Uy, Utheta
    
    def get_M_from_Ki (self,Ki):
        M= np.zeros(len(Ki))
        for k in range(len(Ki)):
            if Ki[k]>=self.mat.C_critic : 
                M[k] = self.mat.EI_max*Ki[k]
            else : 
                M[k] = self.mat.EI_min*Ki[k]
        return M
    
    def run_abaqus(self, Eps, Ki):
        """create fake results of an abaqus calculation"""
        #temporary
        
        N = self.mat.EA*Eps
        M = self.get_M_from_Ki(Ki)
        
        
        return N,M
    
    
        
                             
                            
            
        
        






