# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 19:35:55 2014

@author: jmmauricio
"""

#import xlrd
#from xlwt import Utils
import numpy as np

def range2np(sheet, excel_range):
    if len(excel_range.split(':'))>1:  # se lee un rango
        rango = Utils.cellrange_to_rowcol_pair(excel_range)
        datos = sheet.col_values(rango[1], rango[0], rango[2])
        np_array = np.array(datos)

    if len(excel_range.split(':'))<2: # se lee una celda
        rango = Utils.cell_to_rowcol(excel_range)
        datos = sheet.cell(rango[0], rango[1]).value
        np_array = np.array(datos) 
        
    return np_array
    



class im:
    
    def __init__(self,motor_type='abb_3kW'):
        
        self.motor_type = motor_type
        self.set_motor(self.motor_type)

    def set_motor(self,motor_type):
        self.motor_type = motor_type
        
        self.library = {
            'abb_3kW':  {'P_n': 3e3,'U_n': 400.0, 'R_1': 1.86, 'X_1': 2.2, 'R_f': 1223.0, 'X_mu': 68.0, 
                         'R_2_nom': 1.59, 'X_2_nom': 1.54, 'R_2_start': 1.24, 'X_2_start': 2.2, 
                         'R_2_max': 1.24, 'X_2_max': 1.54, 'n_nom': 1445.0, 'n_max': 1055.0, 'freq_nom':50.0, 
                         'N_pp':2, 'I_nom':6.3},
            'abb_22kW': {'P_n': 22e3,'U_n': 400.0, 'R_1': 0.16, 'X_1': 0.37, 'R_f': 369.0, 'X_mu': 16.6, 
                         'R_2_nom':   0.15, 'X_2_nom':   0.76, 
                         'R_2_start': 0.24, 'X_2_start': 0.37, 
                         'R_2_max':   0.13, 'X_2_max':   0.69, 
                         'n_nom': 1463.0, 'n_max': 1344, 'freq_nom':50.0, 'N_pp':2, 'I_nom':40.7},
            'abb_22kWw': {'P_n': 22e3,'U_n': 400.0, 'R_1': 0.16, 'X_1': 0.37, 'R_f': 369.0, 'X_mu': 16.6, 
                          'R_2_nom':   0.15, 'X_2_nom':   0.76, 
                          'R_2_start': 0.15, 'X_2_start': 0.76, 
                          'R_2_max':   0.15, 'X_2_max':   0.76, 'n_nom': 1463.0, 'n_max': 1344, 'freq_nom':50.0, 
                          'N_pp':2, 'I_nom':40.7},
            'abb_90kW':  {'P_n': 90e3,'U_n': 400.0, 'R_1': 0.027, 'X_1': 0.086, 'R_f': 115.0, 'X_mu': 3.7, 
                          'R_2_nom': 0.0237, 'X_2_nom': 0.19, 'R_2_start': 0.0646, 'X_2_start': 0.086, 
                          'R_2_max': 0.0216, 'X_2_max': 0.15, 'n_nom': 1478.0, 'n_max': 1364, 'freq_nom':50.0, 
                          'N_pp':2, 'I_nom':163}
            } 
            
        self.I_nom  = self.library[self.motor_type]['I_nom']     
        self.N_pp  = self.library[self.motor_type]['N_pp']     
        self.freq_nom = self.library[self.motor_type]['freq_nom']  
        self.freq = self.freq_nom
        self.omega_1 = 2*np.pi*self.freq_nom
        self.Omega_1 = self.omega_1/self.N_pp
        self.n_1=60.0*self.freq_nom/self.N_pp       
        self.N_pp = self.library[self.motor_type]['N_pp']  
        self.P_n = self.library[self.motor_type]['P_n']
        self.R_1 = self.library[self.motor_type]['R_1']
        self.L_1 = self.library[self.motor_type]['X_1']/self.omega_1
        self.R_f = self.library[self.motor_type]['R_f']
        self.L_mu = self.library[self.motor_type]['X_mu']/self.omega_1
        self.R_2_nominal = self.library[self.motor_type]['R_2_nom']
        self.L_2_nominal = self.library[self.motor_type]['X_2_nom']/self.omega_1
        self.R_2_start = self.library[self.motor_type]['R_2_start']
        self.L_2_start = self.library[self.motor_type]['X_2_start']/self.omega_1
        self.R_2_max = self.library[self.motor_type]['R_2_max']
        self.L_2_max = self.library[self.motor_type]['X_2_max']/self.omega_1
        self.n_nom = self.library[self.motor_type]['n_nom'] 
        self.n_max = self.library[self.motor_type]['n_max'] 
        self.operating_point = 'nominal' 
        self.Omega_nom = self.n_nom*2*np.pi/60
        self.T_u_nom =  self.P_n/self.Omega_nom    
        
        
        self.model_type = 'interpolated'
        self.U_n = self.library[self.motor_type]['U_n'] 
        self.U_1 = self.U_n
#        self.R_2 = 0.12
#        self.X_2 = 1.06   # Ω         
#
#
#        self.R_2_1 =  0.3533
#        self.R_2_2 = 0.1783    
#        
#        self.X_2_1 = 0.3740
#        self.X_2_2 = 2.3220   
#        
#        
#        self.n = 1470.0
#        
#        
#        self.T_u_n = 144.0
#        self.n_n = 1463.0
#        
#        self.R_2_nominal, self.X_2_nominal = 0.12, 1.06   # Ω 
#        self.R_2_1tart, self.X_2_1tart = 0.26, 0.35   # Ω 
#        self.Omega_1 = 1500.0*(2.0*np.pi)/60.0
#        self.V_1 = 400.0/np.sqrt(3.0)
#        self.P_n = 22.0e3
#
#        # ABB 22.0 kW 1470.0c rpm
#        self.R_2_nominal, self.X_2_nominal = 0.12, 1.06   # Ω 
#        self.R_2_start, self.X_2_start = 0.26, 0.35   # Ω 
#        
#        self.V_1 = 400.0/np.sqrt(3.0)
#        self.P_n = 22.0e3
#        
#        
#        self.P_1 = 0.0
#        self.P_u = 0.0
#        self.P_2s = 0.0
#        self.P_2r = 0.0
#        self.P_2f = 0.0
#        self.c = 0.0
#        self.c_100 = 0.0
#            
#        self.model_type = 'iyme_giti'
#        self.operating_point = 'nominal'        
#        self.N = np.hstack((np.linspace(1.0,1201.0,20),
#                            np.linspace(1201.0,1400.0,20),
#                            np.linspace(1401.0,1510.0,40)))
#                            
#
#        self.N_2 = []
#        self.T_u_2 = []
#        self.update()
#        #self.update_curve()
        
    def update(self, n_array):
        model_type = self.model_type
        freq = self.freq
        omega_1 = 2*np.pi*freq
        Omega_1 = omega_1/self.N_pp
        
        
        
        n = n_array
        R_1, X_1 = self.R_1, self.L_1*omega_1   # Ω 
        R_f, X_mu = self.R_f, self.L_mu*omega_1 # Ω
        V_1 = self.U_1/np.sqrt(3.0)
            
        Omega = n*(2.0*np.pi)/60.0
        s = (Omega_1 - Omega)/(Omega_1)
        Z_1 = R_1 + 1j*X_1
        Z_m = (R_f * 1j*X_mu) / (R_f + 1j*X_mu)

        if model_type == 'single_cage':
               
            R_2, X_2 = self.R_2, self.X_2   # Ω                                                         
            Z_2 = R_2/s + 1j*X_2          
            Z_mr = Z_m*Z_2/(Z_m + Z_2)
            Z_eq = Z_1 + Z_mr          
            I_1 = V_1/Z_eq    
            S_1 = 3.0*V_1*np.conjugate(I_1)
            E = V_1 - Z_1*I_1          
            I_2 = E/Z_2          
            P_mi = 3.0*R_2*(1.0-s)/s*(np.abs(I_2)**2)
            P_u = P_mi # en el caso del modelo de ABB          
            T_u = P_u/Omega
            self.P_2s = 3*R_1*np.abs(I_1)**2
            self.P_2r = 3*R_2*np.abs(I_2)**2
            self.P_2f = 3*np.abs(E)**2/R_f

        if model_type == 'iyme_giti':
            
            if self.operating_point == 'nominal':
                R_2, X_2 = self.R_2_nominal, self.X_2_nominal   # Ω    
            if self.operating_point == 'start':
                R_2, X_2 = self.R_2_start, self.X_2_start   # Ω                 
            if self.operating_point == 'max':
                R_2, X_2 = self.R_2_max, self.X_2_max   # Ω 
            
            P_mec = (V_1**2/R_f)*2  
            R_fe = 2*R_f

            Z_m = (R_fe * 1j*X_mu) / (R_fe + 1j*X_mu)                                      
            
            if s < 0.001:
                Z_eq = Z_1 + Z_m 
                I_1 = V_1/Z_eq 
                I_2 = 0.0
                P_mi = 0.0
                E = V_1 - Z_1*I_1
                P_u = 0.0
                P_mi = 0.0 
            else:                
                Z_2 = R_2/s + 1j*X_2          
                Z_mr = Z_m*Z_2/(Z_m + Z_2)
                Z_eq = Z_1 + Z_mr 
                I_1 = V_1/Z_eq 
                E = V_1 - Z_1*I_1 
                I_2 = E/Z_2 
                P_mi = 3.0*R_2*(1.0-s)/s*(np.abs(I_2)**2)
                P_u = P_mi - P_mec 
                
            S_1 = 3.0*V_1*np.conjugate(I_1)

                   
            T_u = P_u/Omega
            
            self.P_1 = S_1.real
            self.Q_1 = S_1.imag
            self.P_j1 = 3*R_1*np.abs(I_1)**2
            self.P_j2 = 3*R_2*np.abs(I_2)**2
            self.P_fe = 3*np.abs(E)**2/R_fe
            self.P_mi = P_mi
            self.P_mec = P_mec
            self.R_fe = R_fe           
            self.P_u = P_u

            
        if model_type == 'double_cage':
        
            R_2_1, X_2_1 = self.R_2_1, self.X_2_1   # Ω    
            R_2_2, X_2_2 = self.R_2_2, self.X_2_2   # Ω  
                        
            Z_2_1 = R_2_1/s + 1j*X_2_1
            Z_2_2 = R_2_2/s + 1j*X_2_2
            
            Z_2 = Z_2_1*Z_2_2/(Z_2_1+Z_2_2)
            
            Z_mr = Z_m*Z_2/(Z_m + Z_2)
            Z_eq = Z_1 + Z_mr
            
            S_1 = 3.0*V_1*np.conjugate(I_1)
            I_1 = V_1/Z_eq
            
            E = V_1 - Z_1*I_1
            
            I_2_1 = E/Z_2_1
            I_2_2 = E/Z_2_2
            
            P_mi_1 = 3.0*R_2_1*(1.0-s)/s*(np.abs(I_2_1)**2)
            P_mi_2 = 3.0*R_2_2*(1.0-s)/s*(np.abs(I_2_2)**2)
            
            P_mi = P_mi_1 + P_mi_2 
            P_u = P_mi# en el caso del modelo de ABB
            
            T_u = P_u/Omega
            
        if model_type == 'interpolated':
            slip_n = (self.n_1 - self.n_nom)/self.n_1
            slip_max = (self.n_1 - self.n_max)/self.n_1
            slip_array = np.array([0.0, slip_n,slip_max, 1])
            R_2_array = np.array([self.R_2_nominal,self.R_2_nominal,self.R_2_max,self.R_2_start])
            X_2_array = np.array([self.L_2_nominal,self.L_2_nominal,self.L_2_max,self.L_2_start])*omega_1
            R_2 = np.interp(s,slip_array,R_2_array) # Ω   
            X_2 = np.interp(s,slip_array,X_2_array) # Ω                                                  
            Z_2 = R_2/s + 1j*X_2          
            Z_mr = Z_m*Z_2/(Z_m + Z_2)
            Z_eq = Z_1 + Z_mr          
            I_1 = V_1/Z_eq       
            E = V_1 - Z_1*I_1          
            I_2 = E/Z_2          
            P_a = 3.0*R_2/s*(np.abs(I_2)**2)
            P_j2 = 3*R_2*np.abs(I_2)**2
            P_mi = P_a - P_j2 
            P_u = P_mi # en el caso del modelo de ABB          
            T_mi = P_a/Omega_1
            T_u = T_mi # en el caso del modelo de ABB  
            S_1 = 3.0*V_1*np.conjugate(I_1)
            
            self.R_2_array = R_2
            self.X_2_array = X_2
            self.I_1 = I_1
            self.P_1 = S_1.real
            self.Q_1 = S_1.imag
            self.P_j1 = 3*R_1*np.abs(I_1)**2
            self.P_j2 = 3*R_2*np.abs(I_2)**2
            self.P_fe = 3*np.abs(E)**2/R_f
            self.P_mi = P_mi
            self.P_a = P_a
            self.P_u = P_u  
            self.P_mec = 0.0
            
        self.T_u = T_u  
        self.i_1_m  = np.abs(I_1)
        self.P_u= P_u
        self.P_1= S_1.real
        self.c = self.P_u/self.P_n
        self.c_100 = self.c*100.0
        self.s = s
        
        return T_u
    
#    def update_curve(self):
#        
#        model_type = self.model_type
#        n = self.N
#        R_1, X_1 = self.R_1, self.X_1   # Ω 
#        R_f, X_mu = self.R_f, self.X_mu # Ω
#        n_1 = self.n_1
#        Omega_1 = n_1*(2.0*np.pi)/60.0        
#        V_1 = 400.0/np.sqrt(3.0)
#            
#        Omega = n*(2.0*np.pi)/60.0
#        s = (Omega_1 - Omega)/(Omega_1)
#        Z_1 = R_1 + 1j*X_1
#        Z_m = (R_f * 1j*X_mu) / (R_f + 1j*X_mu)
#
#        if model_type == 'iyme_giti':
#            self.n_0 = n
#            self.update()
#            T_u = self.t_u
#            I_1 = self.i_1_m 
#            
#            
#            
#        if model_type == 'single_cage':
#               
#            R_2, X_2 = self.R_2, self.X_2   # Ω                                                         
#            Z_2 = R_2/s + 1j*X_2          
#            Z_mr = Z_m*Z_2/(Z_m + Z_2)
#            Z_eq = Z_1 + Z_mr          
#            I_1 = V_1/Z_eq       
#            E = V_1 - Z_1*I_1          
#            I_2 = E/Z_2          
#            P_mi = 3.0*R_2*(1.0-s)/s*(np.abs(I_2)**2)
#            P_u = P_mi # en el caso del modelo de ABB          
#            T_u = P_u/Omega
#
#        if model_type == 'double_cage':
#        
#            R_2_1, X_2_1 = self.R_2_1, self.X_2_1   # Ω    
#            R_2_2, X_2_2 = self.R_2_2, self.X_2_2   # Ω  
#                        
#            Z_2_1 = R_2_1/s + 1j*X_2_1
#            Z_2_2 = R_2_2/s + 1j*X_2_2
#            
#            Z_2 = Z_2_1*Z_2_2/(Z_2_1+Z_2_2)
#            
#            Z_mr = Z_m*Z_2/(Z_m + Z_2)
#            Z_eq = Z_1 + Z_mr
#            
#            I_1 = V_1/Z_eq
#            
#            E = V_1 - Z_1*I_1
#            
#            I_2_1 = E/Z_2_1
#            I_2_2 = E/Z_2_2
#            
#            P_mi_1 = 3.0*R_2_1*(1.0-s)/s*(np.abs(I_2_1)**2)
#            P_mi_2 = 3.0*R_2_2*(1.0-s)/s*(np.abs(I_2_2)**2)
#            
#            P_mi = P_mi_1 + P_mi_2 
#            P_u = P_mi# en el caso del modelo de ABB
#            
#            T_u = P_u/Omega
#
#        if model_type == 'interpolated':
#            slip_n = (self.n_1 - self.n_n)/self.n_1
#            slip_array = np.array([1,0.2,slip_n,0.0])
#            R_2_array = np.array([self.R_2_start,self.R_2_max,self.R_2_nominal,self.R_2_nominal])
#            X_2_array = np.array([self.X_2_start,self.X_2_max,self.X_2_nominal,self.X_2_nominal])
#            self.R_2_nominal = self.library[self.motor_type]['R_2_nom']
#            self.X_2_nominal = self.library[self.motor_type]['X_2_nom'] 
#            self.R_2_start = self.library[self.motor_type]['R_2_start']
#            self.X_2_start = self.library[self.motor_type]['X_2_start'] 
#            self.R_2_max = self.library[self.motor_type]['R_2_max']
#            self.X_2_max = self.library[self.motor_type]['X_2_max'] 
#
#            R_2, X_2 = self.R_2, self.X_2   # Ω                                                         
#            Z_2 = R_2/s + 1j*X_2          
#            Z_mr = Z_m*Z_2/(Z_m + Z_2)
#            Z_eq = Z_1 + Z_mr          
#            I_1 = V_1/Z_eq       
#            E = V_1 - Z_1*I_1          
#            I_2 = E/Z_2          
#            P_mi = 3.0*R_2*(1.0-s)/s*(np.abs(I_2)**2)
#            P_u = P_mi # en el caso del modelo de ABB          
#            T_u = P_u/Omega
#        
#        
#        self.T_u  = T_u
#        
#        reduction = rdp(np.vstack((self.N,self.T_u)).T,epsilon=0.2)    
#        
#        self.N_2 = reduction[:,0]
#        self.T_u_2 = reduction[:,1]
#        
#        self.I_1_m  = np.abs(I_1)
#        
#        return T_u
    
    def sankey(self):
        
        import numpy as np
        import matplotlib.pyplot as plt
    
        from matplotlib.sankey import Sankey
        
        P_1_pu = round(self.P_1/self.P_1,5)  
        P_j1_pu = round(self.P_j1/self.P_1,5) 
        P_fe_pu = round(self.P_fe/self.P_1,5) 
        P_j2_pu = round(self.P_j2/self.P_1,5) 
        P_mec_pu = round(self.P_mec/self.P_1,5) 
        P_u_pu = round(self.P_u/self.P_1,3)
        
        s = self.s

        
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[] )
        sankey = Sankey(ax=ax, unit=None)
#        sankey.add(flows=[ P_1_pu], labels=['$\sf P_1$'], orientations=[0])
#        sankey.add(flows=[-P_u_pu], labels=['$\sf P_u$'], orientations=[0])
#        sankey.add(flows=[-P_j1_pu], labels=['$\sf P_{j1}$'], orientations=[-1]) 

        if s < 0.001:
            
            P_1_pu = 1.0  
            P_j1_pu = 1.0 *round(self.P_j1/self.P_1,3) 
            P_fe_pu = 1.0 *round(self.P_fe/self.P_1,3) 
            P_j2_pu = 1.0 *round(self.P_j2/self.P_1,3) 
            P_mec_pu = 1.0 *round(self.P_mec/self.P_1,3) 
            P_u_pu = 1.0 *round(self.P_u/self.P_1,3) 
     
            P_j1_pu = self.P_j1/self.P_1 
            P_fe_pu = self.P_fe/self.P_1 
            P_j2_pu = self.P_j2/self.P_1 
            P_mec_pu = self.P_mec/self.P_1 
            P_u_pu =  self.P_u/self.P_1 

            sankey.add(flows=[P_1_pu,-P_mec_pu,-P_j2_pu,-P_fe_pu,-P_j1_pu],
                   labels=['$\sf P_1$',                      
                           '$\sf P_{mec}$',
                            '$\sf P_{J2}$',  
                            '$\sf P_{Fe}$',
                            '$\sf P_{J1}$'],
                    orientations=   [0,  -1,  -1,  -1, -1],
                    pathlengths = [0.2, 0.1, 0.1, 0.1, 0.1]).finish()

        print(s)
        if s >= 0.001:
            
            P_1_pu = 1.0  
            P_j1_pu = 1.0 *round(self.P_j1/self.P_1,3) 
            P_fe_pu = 1.0 *round(self.P_fe/self.P_1,3) 
            P_j2_pu = 1.0 *round(self.P_j2/self.P_1,3) 
            P_mec_pu = 1.0 *round(self.P_mec/self.P_1,3) 
            P_u_pu = 1.0 *round(self.P_u/self.P_1,3) 
     
            P_j1_pu = self.P_j1/self.P_1 
            P_fe_pu = self.P_fe/self.P_1 
            P_j2_pu = self.P_j2/self.P_1 
            P_mec_pu = self.P_mec/self.P_1 
            P_u_pu =  self.P_u/self.P_1 
            
#        if P_u_pu > 0.0000:
            sankey.add(flows=[P_1_pu,-P_mec_pu,-P_j2_pu,-P_fe_pu,-P_j1_pu,-P_u_pu],
                   labels=['$\sf P_1$',                      
                           '$\sf P_{mec}$',
                            '$\sf P_{J2}$',  
                            '$\sf P_{Fe}$',
                            '$\sf P_{J1}$', 
                           '$\sf P_{u}$'],
                    orientations=[0, -1, -1, -1, -1, 0],
                    pathlengths = [0.2, 0.1, 0.1, 0.1, 0.1, 0.3],).finish()
#
#        if P_u_pu < 10000.00001:                     
#            sankey.add(flows=[P_1_pu,-P_mec_pu,-P_j2_pu,-P_fe_pu,-P_j1_pu],
#                   labels=['$\sf P_1$',                      
#                           '$\sf P_{mec}$',
#                            '$\sf P_{j2}$',  
#                            '$\sf P_{fe}$',
#                            '$\sf P_{j1}$'],
#                    orientations=[0, -1, -1, -1, -1],
#                    pathlengths = [0.2, 0.1, 0.1, 0.1, 0.1],).finish
                
        fig.savefig('im_sankey_{:d}.pdf'.format(int(self.n)))
                      
       
      
        
        
    
if __name__ == "__main__":

    n_array = 1463
    mi_1 = im('abb_22kWw')
    mi_1.update(n_array)
    print(mi_1.T_u)


# To update library using abb motsize and excel file:    
#    import xlrd
#    workbook = xlrd.open_workbook('abb_4poles_3_22_90.xls')
#    ws_22kw_eq = workbook.sheet_by_name(u'1_Eq')
#    
#    ws_eq = ws_22kw_eq
#    
#    mi_1.R_1 =  range2np(ws_eq, 'E20')
#    mi_1.X_1 =  range2np(ws_eq, 'G20')
#    
#    mi_1.R_f =  range2np(ws_eq, 'G21')
#    mi_1.X_mu = range2np(ws_eq, 'E21')
#        
#    mi_1.R_2_nom = range2np(ws_eq, 'G22')
#    mi_1.X_2_nom = range2np(ws_eq, 'E22')        
#    
#    mi_1.R_2_start = range2np(ws_eq, 'G23')
#    mi_1.X_2_start = range2np(ws_eq, 'E23')    
#        
#    mi_1.R_2_max = range2np(ws_eq, 'G24')
#    mi_1.X_2_max = range2np(ws_eq, 'E24')  
#    
#    mi_1.n_nom = range2np(ws_eq, 'E17') 
#
#    motors = {'abb_3kW':{'R_1':float(mi_1.R_1),'X_1':float(mi_1.X_1),
#                         'R_fe':float(mi_1.R_f),'X_mu':float(mi_1.X_mu),
#                         'R_2_nom':float(mi_1.R_2_nom),'X_2_nom':float(mi_1.X_2_nom),
#                         'R_2_start':float(mi_1.R_2_start),'X_2_start':float(mi_1.X_2_start),
#                         'R_2_max':float(mi_1.R_2_max),'X_2_max':float(mi_1.X_2_max),
#                         'n_nom':float(mi_1.n_nom)
#                         }}     
#    print(motors)

'''
    Equivalent motor Volt/phase		231 V	R1s [Ohms]	0.16	X1s [Ohms]	0.37
			                                   Xmagnetizing [Ohms]	16.6	Rfriction+iron [Ohms]	369
			                                   X2 r nom [Ohms]	0.76	R2 r nom [Ohms]	0.15
			                                   X2 start [Ohms]	0.37	R2 start [Ohms]	0.24
                                                  X2 max [Ohms]	0.69	R2 max [Ohms]	0.13
'''
    
 
