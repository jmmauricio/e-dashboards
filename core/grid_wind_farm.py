import numpy as np
import numba
import scipy.optimize as sopt
import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 


class grid_wind_farm_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 1
        self.N_y = 72 
        self.N_z = 27 
        self.N_store = 10000 
        self.params_list = [] 
        self.params_values_list  = [] 
        self.inputs_ini_list = ['v_GRID_a_r', 'v_GRID_a_i', 'v_GRID_b_r', 'v_GRID_b_i', 'v_GRID_c_r', 'v_GRID_c_i', 'i_POI_a_r', 'i_POI_a_i', 'i_POI_b_r', 'i_POI_b_i', 'i_POI_c_r', 'i_POI_c_i', 'i_W1mv_a_r', 'i_W1mv_a_i', 'i_W1mv_b_r', 'i_W1mv_b_i', 'i_W1mv_c_r', 'i_W1mv_c_i', 'i_W2mv_a_r', 'i_W2mv_a_i', 'i_W2mv_b_r', 'i_W2mv_b_i', 'i_W2mv_c_r', 'i_W2mv_c_i', 'i_W3mv_a_r', 'i_W3mv_a_i', 'i_W3mv_b_r', 'i_W3mv_b_i', 'i_W3mv_c_r', 'i_W3mv_c_i', 'p_W1lv_a', 'q_W1lv_a', 'p_W1lv_b', 'q_W1lv_b', 'p_W1lv_c', 'q_W1lv_c', 'p_W2lv_a', 'q_W2lv_a', 'p_W2lv_b', 'q_W2lv_b', 'p_W2lv_c', 'q_W2lv_c', 'p_W3lv_a', 'q_W3lv_a', 'p_W3lv_b', 'q_W3lv_b', 'p_W3lv_c', 'q_W3lv_c', 'p_POImv_a', 'q_POImv_a', 'p_POImv_b', 'q_POImv_b', 'p_POImv_c', 'q_POImv_c'] 
        self.inputs_ini_values_list  = [32999.89801120604, 19052.499999999996, -32999.89801120604, 19052.499999999996, -6.999774942226484e-12, -38105.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.8078793573295115e-06, 7.375024324574042e-08, 0.0, 0.0, 0.0, 0.0, 666666.6675282058, 0.0034586082911118865, 666666.6676866676, 0.0034682357509154826, 666666.6675938459, 0.0035714048135560006, 666666.6675202895, 0.0034475086722522974, 666666.667678176, 0.00345750994165428, 666666.6675852421, 0.0035600910341599956, 666666.6675044703, 0.003425316303037107, 666666.6676612026, 0.003436064056586474, 666666.6675680466, 0.0035374678845983, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0] 
        self.inputs_run_list = ['v_GRID_a_r', 'v_GRID_a_i', 'v_GRID_b_r', 'v_GRID_b_i', 'v_GRID_c_r', 'v_GRID_c_i', 'i_POI_a_r', 'i_POI_a_i', 'i_POI_b_r', 'i_POI_b_i', 'i_POI_c_r', 'i_POI_c_i', 'i_W1mv_a_r', 'i_W1mv_a_i', 'i_W1mv_b_r', 'i_W1mv_b_i', 'i_W1mv_c_r', 'i_W1mv_c_i', 'i_W2mv_a_r', 'i_W2mv_a_i', 'i_W2mv_b_r', 'i_W2mv_b_i', 'i_W2mv_c_r', 'i_W2mv_c_i', 'i_W3mv_a_r', 'i_W3mv_a_i', 'i_W3mv_b_r', 'i_W3mv_b_i', 'i_W3mv_c_r', 'i_W3mv_c_i', 'p_W1lv_a', 'q_W1lv_a', 'p_W1lv_b', 'q_W1lv_b', 'p_W1lv_c', 'q_W1lv_c', 'p_W2lv_a', 'q_W2lv_a', 'p_W2lv_b', 'q_W2lv_b', 'p_W2lv_c', 'q_W2lv_c', 'p_W3lv_a', 'q_W3lv_a', 'p_W3lv_b', 'q_W3lv_b', 'p_W3lv_c', 'q_W3lv_c', 'p_POImv_a', 'q_POImv_a', 'p_POImv_b', 'q_POImv_b', 'p_POImv_c', 'q_POImv_c'] 
        self.inputs_run_values_list = [32999.89801120604, 19052.499999999996, -32999.89801120604, 19052.499999999996, -6.999774942226484e-12, -38105.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.8078793573295115e-06, 7.375024324574042e-08, 0.0, 0.0, 0.0, 0.0, 666666.6675282058, 0.0034586082911118865, 666666.6676866676, 0.0034682357509154826, 666666.6675938459, 0.0035714048135560006, 666666.6675202895, 0.0034475086722522974, 666666.667678176, 0.00345750994165428, 666666.6675852421, 0.0035600910341599956, 666666.6675044703, 0.003425316303037107, 666666.6676612026, 0.003436064056586474, 666666.6675680466, 0.0035374678845983, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0] 
        self.outputs_list = ['v_GRID_a_m', 'v_GRID_b_m', 'v_GRID_c_m', 'v_W1lv_a_m', 'v_W1lv_b_m', 'v_W1lv_c_m', 'v_W2lv_a_m', 'v_W2lv_b_m', 'v_W2lv_c_m', 'v_W3lv_a_m', 'v_W3lv_b_m', 'v_W3lv_c_m', 'v_POImv_a_m', 'v_POImv_b_m', 'v_POImv_c_m', 'v_POI_a_m', 'v_POI_b_m', 'v_POI_c_m', 'v_W1mv_a_m', 'v_W1mv_b_m', 'v_W1mv_c_m', 'v_W2mv_a_m', 'v_W2mv_b_m', 'v_W2mv_c_m', 'v_W3mv_a_m', 'v_W3mv_b_m', 'v_W3mv_c_m'] 
        self.x_list = ['a'] 
        self.y_run_list = ['v_W1lv_a_r', 'v_W1lv_a_i', 'v_W1lv_b_r', 'v_W1lv_b_i', 'v_W1lv_c_r', 'v_W1lv_c_i', 'v_W2lv_a_r', 'v_W2lv_a_i', 'v_W2lv_b_r', 'v_W2lv_b_i', 'v_W2lv_c_r', 'v_W2lv_c_i', 'v_W3lv_a_r', 'v_W3lv_a_i', 'v_W3lv_b_r', 'v_W3lv_b_i', 'v_W3lv_c_r', 'v_W3lv_c_i', 'v_POImv_a_r', 'v_POImv_a_i', 'v_POImv_b_r', 'v_POImv_b_i', 'v_POImv_c_r', 'v_POImv_c_i', 'v_POI_a_r', 'v_POI_a_i', 'v_POI_b_r', 'v_POI_b_i', 'v_POI_c_r', 'v_POI_c_i', 'v_W1mv_a_r', 'v_W1mv_a_i', 'v_W1mv_b_r', 'v_W1mv_b_i', 'v_W1mv_c_r', 'v_W1mv_c_i', 'v_W2mv_a_r', 'v_W2mv_a_i', 'v_W2mv_b_r', 'v_W2mv_b_i', 'v_W2mv_c_r', 'v_W2mv_c_i', 'v_W3mv_a_r', 'v_W3mv_a_i', 'v_W3mv_b_r', 'v_W3mv_b_i', 'v_W3mv_c_r', 'v_W3mv_c_i', 'i_W1lv_a_r', 'i_W1lv_a_i', 'i_W1lv_b_r', 'i_W1lv_b_i', 'i_W1lv_c_r', 'i_W1lv_c_i', 'i_W2lv_a_r', 'i_W2lv_a_i', 'i_W2lv_b_r', 'i_W2lv_b_i', 'i_W2lv_c_r', 'i_W2lv_c_i', 'i_W3lv_a_r', 'i_W3lv_a_i', 'i_W3lv_b_r', 'i_W3lv_b_i', 'i_W3lv_c_r', 'i_W3lv_c_i', 'i_POImv_a_r', 'i_POImv_a_i', 'i_POImv_b_r', 'i_POImv_b_i', 'i_POImv_c_r', 'i_POImv_c_i'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['v_W1lv_a_r', 'v_W1lv_a_i', 'v_W1lv_b_r', 'v_W1lv_b_i', 'v_W1lv_c_r', 'v_W1lv_c_i', 'v_W2lv_a_r', 'v_W2lv_a_i', 'v_W2lv_b_r', 'v_W2lv_b_i', 'v_W2lv_c_r', 'v_W2lv_c_i', 'v_W3lv_a_r', 'v_W3lv_a_i', 'v_W3lv_b_r', 'v_W3lv_b_i', 'v_W3lv_c_r', 'v_W3lv_c_i', 'v_POImv_a_r', 'v_POImv_a_i', 'v_POImv_b_r', 'v_POImv_b_i', 'v_POImv_c_r', 'v_POImv_c_i', 'v_POI_a_r', 'v_POI_a_i', 'v_POI_b_r', 'v_POI_b_i', 'v_POI_c_r', 'v_POI_c_i', 'v_W1mv_a_r', 'v_W1mv_a_i', 'v_W1mv_b_r', 'v_W1mv_b_i', 'v_W1mv_c_r', 'v_W1mv_c_i', 'v_W2mv_a_r', 'v_W2mv_a_i', 'v_W2mv_b_r', 'v_W2mv_b_i', 'v_W2mv_c_r', 'v_W2mv_c_i', 'v_W3mv_a_r', 'v_W3mv_a_i', 'v_W3mv_b_r', 'v_W3mv_b_i', 'v_W3mv_c_r', 'v_W3mv_c_i', 'i_W1lv_a_r', 'i_W1lv_a_i', 'i_W1lv_b_r', 'i_W1lv_b_i', 'i_W1lv_c_r', 'i_W1lv_c_i', 'i_W2lv_a_r', 'i_W2lv_a_i', 'i_W2lv_b_r', 'i_W2lv_b_i', 'i_W2lv_c_r', 'i_W2lv_c_i', 'i_W3lv_a_r', 'i_W3lv_a_i', 'i_W3lv_b_r', 'i_W3lv_b_i', 'i_W3lv_c_r', 'i_W3lv_c_i', 'i_POImv_a_r', 'i_POImv_a_i', 'i_POImv_b_r', 'i_POImv_b_i', 'i_POImv_c_r', 'i_POImv_c_i'] 
        self.xy_ini_list = self.x_list + self.y_ini_list 
        self.t = 0.0
        self.it = 0
        self.it_store = 0
        self.xy_prev = np.zeros((self.N_x+self.N_y,1))
        self.initialization_tol = 1e-6
        self.N_u = len(self.inputs_run_list) 
        self.sopt_root_method='hybr'
        self.sopt_root_jac=True
        self.u_ini_list = self.inputs_ini_list
        self.u_ini_values_list = self.inputs_ini_values_list
        self.u_run_list = self.inputs_run_list
        self.u_run_values_list = self.inputs_run_values_list
        
        self.update() 


    def update(self): 

        self.N_steps = int(np.ceil(self.t_end/self.Dt)) 
        dt = [  
              ('t_end', np.float64),
              ('Dt', np.float64),
              ('decimation', np.float64),
              ('itol', np.float64),
              ('Dt_max', np.float64),
              ('Dt_min', np.float64),
              ('solvern', np.int64),
              ('imax', np.int64),
              ('N_steps', np.int64),
              ('N_store', np.int64),
              ('N_x', np.int64),
              ('N_y', np.int64),
              ('N_z', np.int64),
              ('t', np.float64),
              ('it', np.int64),
              ('it_store', np.int64),
              ('idx', np.int64),
              ('idy', np.int64),
              ('f', np.float64, (self.N_x,1)),
              ('x', np.float64, (self.N_x,1)),
              ('x_0', np.float64, (self.N_x,1)),
              ('g', np.float64, (self.N_y,1)),
              ('y_run', np.float64, (self.N_y,1)),
              ('y_ini', np.float64, (self.N_y,1)),
              ('y_0', np.float64, (self.N_y,1)),
              ('h', np.float64, (self.N_z,1)),
              ('Fx', np.float64, (self.N_x,self.N_x)),
              ('Fy', np.float64, (self.N_x,self.N_y)),
              ('Gx', np.float64, (self.N_y,self.N_x)),
              ('Gy', np.float64, (self.N_y,self.N_y)),
              ('Fu', np.float64, (self.N_x,self.N_u)),
              ('Gu', np.float64, (self.N_y,self.N_u)),
              ('Hx', np.float64, (self.N_z,self.N_x)),
              ('Hy', np.float64, (self.N_z,self.N_y)),
              ('Hu', np.float64, (self.N_z,self.N_u)),
              ('Fx_ini', np.float64, (self.N_x,self.N_x)),
              ('Fy_ini', np.float64, (self.N_x,self.N_y)),
              ('Gx_ini', np.float64, (self.N_y,self.N_x)),
              ('Gy_ini', np.float64, (self.N_y,self.N_y)),
              ('T', np.float64, (self.N_store+1,1)),
              ('X', np.float64, (self.N_store+1,self.N_x)),
              ('Y', np.float64, (self.N_store+1,self.N_y)),
              ('Z', np.float64, (self.N_store+1,self.N_z)),
              ('iters', np.float64, (self.N_store+1,1)),
             ]

        values = [
                self.t_end,                          
                self.Dt,
                self.decimation,
                self.itol,
                self.Dt_max,
                self.Dt_min,
                self.solvern,
                self.imax,
                self.N_steps,
                self.N_store,
                self.N_x,
                self.N_y,
                self.N_z,
                self.t,
                self.it,
                self.it_store,
                0,                                     # idx
                0,                                     # idy
                np.zeros((self.N_x,1)),                # f
                np.zeros((self.N_x,1)),                # x
                np.zeros((self.N_x,1)),                # x_0
                np.zeros((self.N_y,1)),                # g
                np.zeros((self.N_y,1)),                # y_run
                np.zeros((self.N_y,1)),                # y_ini
                np.zeros((self.N_y,1)),                # y_0
                np.zeros((self.N_z,1)),                # h
                np.zeros((self.N_x,self.N_x)),         # Fx   
                np.zeros((self.N_x,self.N_y)),         # Fy 
                np.zeros((self.N_y,self.N_x)),         # Gx 
                np.zeros((self.N_y,self.N_y)),         # Fy
                np.zeros((self.N_x,self.N_u)),         # Fu 
                np.zeros((self.N_y,self.N_u)),         # Gu 
                np.zeros((self.N_z,self.N_x)),         # Hx 
                np.zeros((self.N_z,self.N_y)),         # Hy 
                np.zeros((self.N_z,self.N_u)),         # Hu 
                np.zeros((self.N_x,self.N_x)),         # Fx_ini  
                np.zeros((self.N_x,self.N_y)),         # Fy_ini 
                np.zeros((self.N_y,self.N_x)),         # Gx_ini 
                np.zeros((self.N_y,self.N_y)),         # Fy_ini 
                np.zeros((self.N_store+1,1)),          # T
                np.zeros((self.N_store+1,self.N_x)),   # X
                np.zeros((self.N_store+1,self.N_y)),   # Y
                np.zeros((self.N_store+1,self.N_z)),   # Z
                np.zeros((self.N_store+1,1)),          # iters
                ]  

        dt += [(item,np.float64) for item in self.params_list]
        values += [item for item in self.params_values_list]

        for item_id,item_val in zip(self.inputs_ini_list,self.inputs_ini_values_list):
            if item_id in self.inputs_run_list: continue
            dt += [(item_id,np.float64)]
            values += [item_val]

        dt += [(item,np.float64) for item in self.inputs_run_list]
        values += [item for item in self.inputs_run_values_list]

        self.struct = np.rec.array([tuple(values)], dtype=np.dtype(dt))

    def load_params(self,data_input):

        if type(data_input) == str:
            json_file = data_input
            self.json_file = json_file
            self.json_data = open(json_file).read().replace("'",'"')
            data = json.loads(self.json_data)
        elif type(data_input) == dict:
            data = data_input

        self.data = data
        for item in self.data:
            self.struct[0][item] = self.data[item]
            self.params_values_list[self.params_list.index(item)] = self.data[item]



    def ini_problem(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        ini(self.struct,2)
        ini(self.struct,3)       
        fg = np.vstack((self.struct[0].f,self.struct[0].g))[:,0]
        return fg

    def run_problem(self,x):
        t = self.struct[0].t
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_run[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        run(t,self.struct,2)
        run(t,self.struct,3)
        run(t,self.struct,10)
        run(t,self.struct,11)
        run(t,self.struct,12)
        run(t,self.struct,13)
        
        fg = np.vstack((self.struct[0].f,self.struct[0].g))[:,0]
        return fg
    

    def run_dae_jacobian(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_run[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        run(0.0,self.struct,10)
        run(0.0,self.struct,11)     
        run(0.0,self.struct,12)
        run(0.0,self.struct,13)
        A_c = np.block([[self.struct[0].Fx,self.struct[0].Fy],
                        [self.struct[0].Gx,self.struct[0].Gy]])
        return A_c
    
    def eval_jacobians(self):

        run(0.0,self.struct,10)
        run(0.0,self.struct,11)  
        run(0.0,self.struct,12) 

        return 1


    def ini_dae_jacobian(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        ini(self.struct,10)
        ini(self.struct,11)       
        A_c = np.block([[self.struct[0].Fx_ini,self.struct[0].Fy_ini],
                        [self.struct[0].Gx_ini,self.struct[0].Gy_ini]])
        return A_c



    def f_ode(self,x):
        self.struct[0].x[:,0] = x
        run(self.struct,1)
        return self.struct[0].f[:,0]

    def f_odeint(self,x,t):
        self.struct[0].x[:,0] = x
        run(self.struct,1)
        return self.struct[0].f[:,0]

    def f_ivp(self,t,x):
        self.struct[0].x[:,0] = x
        run(self.struct,1)
        return self.struct[0].f[:,0]

    def Fx_ode(self,x):
        self.struct[0].x[:,0] = x
        run(self.struct,10)
        return self.struct[0].Fx

    def eval_A(self):
        
        Fx = self.struct[0].Fx
        Fy = self.struct[0].Fy
        Gx = self.struct[0].Gx
        Gy = self.struct[0].Gy
        
        A = Fx - Fy @ np.linalg.solve(Gy,Gx)
        
        self.A = A
        
        return A

    def eval_A_ini(self):
        
        Fx = self.struct[0].Fx_ini
        Fy = self.struct[0].Fy_ini
        Gx = self.struct[0].Gx_ini
        Gy = self.struct[0].Gy_ini
        
        A = Fx - Fy @ np.linalg.solve(Gy,Gx)
        
        
        return A
    
    def reset(self):
        for param,param_value in zip(self.params_list,self.params_values_list):
            self.struct[0][param] = param_value
        for input_name,input_value in zip(self.inputs_ini_list,self.inputs_ini_values_list):
            self.struct[0][input_name] = input_value   
        for input_name,input_value in zip(self.inputs_run_list,self.inputs_run_values_list):
            self.struct[0][input_name] = input_value  

    def simulate(self,events,xy0=0):
        
        # initialize both the ini and the run system
        self.initialize(events,xy0=xy0)
        
        ## solve 
        #daesolver(self.struct)    # run until first event

        # simulation run
        for event in events[1:]:  
            # make all the desired changes
            for item in event:
                self.struct[0][item] = event[item]
            daesolver(self.struct)    # run until next event
            
        
        T,X,Y,Z = self.post()
        
        return T,X,Y,Z
    
    def run(self,events):
        

        # simulation run
        for event in events:  
            # make all the desired changes
            for item in event:
                self.struct[0][item] = event[item]
            daesolver(self.struct)    # run until next event
            
        return 1
    
    
    def post(self):
        
        # post process result    
        T = self.struct[0]['T'][:self.struct[0].it_store]
        X = self.struct[0]['X'][:self.struct[0].it_store,:]
        Y = self.struct[0]['Y'][:self.struct[0].it_store,:]
        Z = self.struct[0]['Z'][:self.struct[0].it_store,:]
        iters = self.struct[0]['iters'][:self.struct[0].it_store,:]
    
        self.T = T
        self.X = X
        self.Y = Y
        self.Z = Z
        self.iters = iters
        
        return T,X,Y,Z
        
        
    def initialize(self,events=[{}],xy0=0):
        '''
        

        Parameters
        ----------
        events : dictionary 
            Dictionary with at least 't_end' and all inputs and parameters 
            that need to be changed.
        xy0 : float or string, optional
            0 means all states should be zero as initial guess. 
            If not zero all the states initial guess are the given input.
            If 'prev' it uses the last known initialization result as initial guess.

        Returns
        -------
        T : TYPE
            DESCRIPTION.
        X : TYPE
            DESCRIPTION.
        Y : TYPE
            DESCRIPTION.
        Z : TYPE
            DESCRIPTION.

        '''
        # simulation parameters
        self.struct[0].it = 0       # set time step to zero
        self.struct[0].it_store = 0 # set storage to zero
        self.struct[0].t = 0.0      # set time to zero
                    
        # initialization
        it_event = 0
        event = events[it_event]
        for item in event:
            self.struct[0][item] = event[item]
            
        
        ## compute initial conditions using x and y_ini 
        if xy0 == 0:
            xy0 = np.zeros(self.N_x+self.N_y)
        elif xy0 == 1:
            xy0 = np.ones(self.N_x+self.N_y)
        elif xy0 == 'prev':
            xy0 = self.xy_prev
        else:
            xy0 = xy0*np.ones(self.N_x+self.N_y)

        #xy = sopt.fsolve(self.ini_problem,xy0, jac=self.ini_dae_jacobian )
        if self.sopt_root_jac:
            sol = sopt.root(self.ini_problem, xy0, 
                            jac=self.ini_dae_jacobian, 
                            method=self.sopt_root_method, tol=self.initialization_tol)
        else:
            sol = sopt.root(self.ini_problem, xy0, method=self.sopt_root_method)

        self.initialization_ok = True
        if sol.success == False:
            print('initialization not found!')
            self.initialization_ok = False

            T = self.struct[0]['T'][:self.struct[0].it_store]
            X = self.struct[0]['X'][:self.struct[0].it_store,:]
            Y = self.struct[0]['Y'][:self.struct[0].it_store,:]
            Z = self.struct[0]['Z'][:self.struct[0].it_store,:]
            iters = self.struct[0]['iters'][:self.struct[0].it_store,:]

        if self.initialization_ok:
            xy = sol.x
            self.xy_prev = xy
            self.struct[0].x[:,0] = xy[0:self.N_x]
            self.struct[0].y_run[:,0] = xy[self.N_x:]

            ## y_ini to u_run
            for item in self.inputs_run_list:
                if item in self.y_ini_list:
                    self.struct[0][item] = self.struct[0].y_ini[self.y_ini_list.index(item)]

            ## u_ini to y_run
            for item in self.inputs_ini_list:
                if item in self.y_run_list:
                    self.struct[0].y_run[self.y_run_list.index(item)] = self.struct[0][item]


            #xy = sopt.fsolve(self.ini_problem,xy0, jac=self.ini_dae_jacobian )
            if self.sopt_root_jac:
                sol = sopt.root(self.run_problem, xy0, 
                                jac=self.run_dae_jacobian, 
                                method=self.sopt_root_method, tol=self.initialization_tol)
            else:
                sol = sopt.root(self.run_problem, xy0, method=self.sopt_root_method)

            # evaluate f and g
            run(0.0,self.struct,2)
            run(0.0,self.struct,3)                

            
            # evaluate run jacobians 
            run(0.0,self.struct,10)
            run(0.0,self.struct,11)                
            run(0.0,self.struct,12) 
            run(0.0,self.struct,14) 
             
            # post process result    
            T = self.struct[0]['T'][:self.struct[0].it_store]
            X = self.struct[0]['X'][:self.struct[0].it_store,:]
            Y = self.struct[0]['Y'][:self.struct[0].it_store,:]
            Z = self.struct[0]['Z'][:self.struct[0].it_store,:]
            iters = self.struct[0]['iters'][:self.struct[0].it_store,:]
        
            self.T = T
            self.X = X
            self.Y = Y
            self.Z = Z
            self.iters = iters
            
        return self.initialization_ok
    
    
    def get_value(self,name):
        if name in self.inputs_run_list:
            value = self.struct[0][name]
        if name in self.x_list:
            idx = self.x_list.index(name)
            value = self.struct[0].x[idx,0]
        if name in self.y_run_list:
            idy = self.y_run_list.index(name)
            value = self.struct[0].y_run[idy,0]
        if name in self.params_list:
            value = self.struct[0][name]
        if name in self.outputs_list:
            value = self.struct[0].h[self.outputs_list.index(name),0] 

        return value
    
    def get_values(self,name):
        if name in self.x_list:
            values = self.X[:,self.x_list.index(name)]
        if name in self.y_run_list:
            values = self.Y[:,self.y_run_list.index(name)]
        if name in self.outputs_list:
            values = self.Z[:,self.outputs_list.index(name)]
                        
        return values

    def get_mvalue(self,names):
        '''

        Parameters
        ----------
        names : list
            list of variables names to return each value.

        Returns
        -------
        mvalue : TYPE
            list of value of each variable.

        '''
        mvalue = []
        for name in names:
            mvalue += [self.get_value(name)]
                        
        return mvalue
    
    def set_value(self,name,value):
        if name in self.inputs_run_list:
            self.struct[0][name] = value
        if name in self.params_list:
            self.struct[0][name] = value
            
    def report_x(self,value_format='5.2f'):
        for item in self.x_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')

    def report_y(self,value_format='5.2f'):
        for item in self.y_run_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')
            
    def report_u(self,value_format='5.2f'):
        for item in self.inputs_run_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')

    def report_z(self,value_format='5.2f'):
        for item in self.outputs_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')
            
    def get_x(self):
        return self.struct[0].x


@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    
    # Inputs:
    v_GRID_a_r = struct[0].v_GRID_a_r
    v_GRID_a_i = struct[0].v_GRID_a_i
    v_GRID_b_r = struct[0].v_GRID_b_r
    v_GRID_b_i = struct[0].v_GRID_b_i
    v_GRID_c_r = struct[0].v_GRID_c_r
    v_GRID_c_i = struct[0].v_GRID_c_i
    i_POI_a_r = struct[0].i_POI_a_r
    i_POI_a_i = struct[0].i_POI_a_i
    i_POI_b_r = struct[0].i_POI_b_r
    i_POI_b_i = struct[0].i_POI_b_i
    i_POI_c_r = struct[0].i_POI_c_r
    i_POI_c_i = struct[0].i_POI_c_i
    i_W1mv_a_r = struct[0].i_W1mv_a_r
    i_W1mv_a_i = struct[0].i_W1mv_a_i
    i_W1mv_b_r = struct[0].i_W1mv_b_r
    i_W1mv_b_i = struct[0].i_W1mv_b_i
    i_W1mv_c_r = struct[0].i_W1mv_c_r
    i_W1mv_c_i = struct[0].i_W1mv_c_i
    i_W2mv_a_r = struct[0].i_W2mv_a_r
    i_W2mv_a_i = struct[0].i_W2mv_a_i
    i_W2mv_b_r = struct[0].i_W2mv_b_r
    i_W2mv_b_i = struct[0].i_W2mv_b_i
    i_W2mv_c_r = struct[0].i_W2mv_c_r
    i_W2mv_c_i = struct[0].i_W2mv_c_i
    i_W3mv_a_r = struct[0].i_W3mv_a_r
    i_W3mv_a_i = struct[0].i_W3mv_a_i
    i_W3mv_b_r = struct[0].i_W3mv_b_r
    i_W3mv_b_i = struct[0].i_W3mv_b_i
    i_W3mv_c_r = struct[0].i_W3mv_c_r
    i_W3mv_c_i = struct[0].i_W3mv_c_i
    p_W1lv_a = struct[0].p_W1lv_a
    q_W1lv_a = struct[0].q_W1lv_a
    p_W1lv_b = struct[0].p_W1lv_b
    q_W1lv_b = struct[0].q_W1lv_b
    p_W1lv_c = struct[0].p_W1lv_c
    q_W1lv_c = struct[0].q_W1lv_c
    p_W2lv_a = struct[0].p_W2lv_a
    q_W2lv_a = struct[0].q_W2lv_a
    p_W2lv_b = struct[0].p_W2lv_b
    q_W2lv_b = struct[0].q_W2lv_b
    p_W2lv_c = struct[0].p_W2lv_c
    q_W2lv_c = struct[0].q_W2lv_c
    p_W3lv_a = struct[0].p_W3lv_a
    q_W3lv_a = struct[0].q_W3lv_a
    p_W3lv_b = struct[0].p_W3lv_b
    q_W3lv_b = struct[0].q_W3lv_b
    p_W3lv_c = struct[0].p_W3lv_c
    q_W3lv_c = struct[0].q_W3lv_c
    p_POImv_a = struct[0].p_POImv_a
    q_POImv_a = struct[0].q_POImv_a
    p_POImv_b = struct[0].p_POImv_b
    q_POImv_b = struct[0].q_POImv_b
    p_POImv_c = struct[0].p_POImv_c
    q_POImv_c = struct[0].q_POImv_c
    
    # Dynamical states:
    a = struct[0].x[0,0]
    
    # Algebraic states:
    v_W1lv_a_r = struct[0].y_run[0,0]
    v_W1lv_a_i = struct[0].y_run[1,0]
    v_W1lv_b_r = struct[0].y_run[2,0]
    v_W1lv_b_i = struct[0].y_run[3,0]
    v_W1lv_c_r = struct[0].y_run[4,0]
    v_W1lv_c_i = struct[0].y_run[5,0]
    v_W2lv_a_r = struct[0].y_run[6,0]
    v_W2lv_a_i = struct[0].y_run[7,0]
    v_W2lv_b_r = struct[0].y_run[8,0]
    v_W2lv_b_i = struct[0].y_run[9,0]
    v_W2lv_c_r = struct[0].y_run[10,0]
    v_W2lv_c_i = struct[0].y_run[11,0]
    v_W3lv_a_r = struct[0].y_run[12,0]
    v_W3lv_a_i = struct[0].y_run[13,0]
    v_W3lv_b_r = struct[0].y_run[14,0]
    v_W3lv_b_i = struct[0].y_run[15,0]
    v_W3lv_c_r = struct[0].y_run[16,0]
    v_W3lv_c_i = struct[0].y_run[17,0]
    v_POImv_a_r = struct[0].y_run[18,0]
    v_POImv_a_i = struct[0].y_run[19,0]
    v_POImv_b_r = struct[0].y_run[20,0]
    v_POImv_b_i = struct[0].y_run[21,0]
    v_POImv_c_r = struct[0].y_run[22,0]
    v_POImv_c_i = struct[0].y_run[23,0]
    v_POI_a_r = struct[0].y_run[24,0]
    v_POI_a_i = struct[0].y_run[25,0]
    v_POI_b_r = struct[0].y_run[26,0]
    v_POI_b_i = struct[0].y_run[27,0]
    v_POI_c_r = struct[0].y_run[28,0]
    v_POI_c_i = struct[0].y_run[29,0]
    v_W1mv_a_r = struct[0].y_run[30,0]
    v_W1mv_a_i = struct[0].y_run[31,0]
    v_W1mv_b_r = struct[0].y_run[32,0]
    v_W1mv_b_i = struct[0].y_run[33,0]
    v_W1mv_c_r = struct[0].y_run[34,0]
    v_W1mv_c_i = struct[0].y_run[35,0]
    v_W2mv_a_r = struct[0].y_run[36,0]
    v_W2mv_a_i = struct[0].y_run[37,0]
    v_W2mv_b_r = struct[0].y_run[38,0]
    v_W2mv_b_i = struct[0].y_run[39,0]
    v_W2mv_c_r = struct[0].y_run[40,0]
    v_W2mv_c_i = struct[0].y_run[41,0]
    v_W3mv_a_r = struct[0].y_run[42,0]
    v_W3mv_a_i = struct[0].y_run[43,0]
    v_W3mv_b_r = struct[0].y_run[44,0]
    v_W3mv_b_i = struct[0].y_run[45,0]
    v_W3mv_c_r = struct[0].y_run[46,0]
    v_W3mv_c_i = struct[0].y_run[47,0]
    i_W1lv_a_r = struct[0].y_run[48,0]
    i_W1lv_a_i = struct[0].y_run[49,0]
    i_W1lv_b_r = struct[0].y_run[50,0]
    i_W1lv_b_i = struct[0].y_run[51,0]
    i_W1lv_c_r = struct[0].y_run[52,0]
    i_W1lv_c_i = struct[0].y_run[53,0]
    i_W2lv_a_r = struct[0].y_run[54,0]
    i_W2lv_a_i = struct[0].y_run[55,0]
    i_W2lv_b_r = struct[0].y_run[56,0]
    i_W2lv_b_i = struct[0].y_run[57,0]
    i_W2lv_c_r = struct[0].y_run[58,0]
    i_W2lv_c_i = struct[0].y_run[59,0]
    i_W3lv_a_r = struct[0].y_run[60,0]
    i_W3lv_a_i = struct[0].y_run[61,0]
    i_W3lv_b_r = struct[0].y_run[62,0]
    i_W3lv_b_i = struct[0].y_run[63,0]
    i_W3lv_c_r = struct[0].y_run[64,0]
    i_W3lv_c_i = struct[0].y_run[65,0]
    i_POImv_a_r = struct[0].y_run[66,0]
    i_POImv_a_i = struct[0].y_run[67,0]
    i_POImv_b_r = struct[0].y_run[68,0]
    i_POImv_b_i = struct[0].y_run[69,0]
    i_POImv_c_r = struct[0].y_run[70,0]
    i_POImv_c_i = struct[0].y_run[71,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = 1 - a
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -0.127279074345343*i_POI_a_i + 0.076409970853885*i_POI_a_r + 0.0636395371726705*i_POI_b_i - 0.038204985426942*i_POI_b_r + 0.0636395371726721*i_POI_c_i - 0.0382049854269431*i_POI_c_r - 0.0971901504394946*i_POImv_a_i + 0.0280418380652321*i_POImv_a_r + 0.0971901504394895*i_POImv_c_i - 0.0280418380652375*i_POImv_c_r - 0.0154231877861473*i_W1lv_a_i + 0.0031739357845936*i_W1lv_a_r + 0.00199839389307364*i_W1lv_b_i - 0.000634767892296793*i_W1lv_b_r + 0.00199839389307368*i_W1lv_c_i - 0.000634767892296814*i_W1lv_c_r - 0.100328108879392*i_W1mv_a_i + 0.0318681229122137*i_W1mv_a_r + 0.100328108879387*i_W1mv_c_i - 0.0318681229122191*i_W1mv_c_r - 0.00395512560257793*i_W2lv_a_i + 0.00121874317417018*i_W2lv_a_r + 0.00197756280128894*i_W2lv_b_i - 0.00060937158708508*i_W2lv_b_r + 0.00197756280128899*i_W2lv_c_i - 0.000609371587085102*i_W2lv_c_r - 0.0992822970142266*i_W2mv_a_i + 0.0305931173773959*i_W2mv_a_r + 0.0992822970142214*i_W2mv_c_i - 0.0305931173774017*i_W2mv_c_r - 0.00391345649507333*i_W3lv_a_i + 0.00116793362771941*i_W3lv_a_r + 0.00195672824753664*i_W3lv_b_i - 0.000583966813859693*i_W3lv_b_r + 0.00195672824753669*i_W3lv_c_i - 0.000583966813859714*i_W3lv_c_r - 0.0982363113431537*i_W3mv_a_i + 0.0293176867112763*i_W3mv_a_r + 0.0982363113431486*i_W3mv_c_i - 0.0293176867112818*i_W3mv_c_r + 1.71338624816169e-6*v_GRID_a_i + 0.00697522388628822*v_GRID_a_r - 8.56693124078082e-7*v_GRID_b_i - 0.00348761194314407*v_GRID_b_r - 8.56693124085475e-7*v_GRID_c_i - 0.00348761194314416*v_GRID_c_r - v_W1lv_a_r
        struct[0].g[1,0] = 0.076409970853885*i_POI_a_i + 0.127279074345343*i_POI_a_r - 0.038204985426942*i_POI_b_i - 0.0636395371726705*i_POI_b_r - 0.0382049854269431*i_POI_c_i - 0.0636395371726721*i_POI_c_r + 0.0280418380652321*i_POImv_a_i + 0.0971901504394946*i_POImv_a_r - 0.0280418380652375*i_POImv_c_i - 0.0971901504394895*i_POImv_c_r + 0.0031739357845936*i_W1lv_a_i + 0.0154231877861473*i_W1lv_a_r - 0.000634767892296793*i_W1lv_b_i - 0.00199839389307364*i_W1lv_b_r - 0.000634767892296814*i_W1lv_c_i - 0.00199839389307368*i_W1lv_c_r + 0.0318681229122137*i_W1mv_a_i + 0.100328108879392*i_W1mv_a_r - 0.0318681229122191*i_W1mv_c_i - 0.100328108879387*i_W1mv_c_r + 0.00121874317417018*i_W2lv_a_i + 0.00395512560257793*i_W2lv_a_r - 0.00060937158708508*i_W2lv_b_i - 0.00197756280128894*i_W2lv_b_r - 0.000609371587085102*i_W2lv_c_i - 0.00197756280128899*i_W2lv_c_r + 0.0305931173773959*i_W2mv_a_i + 0.0992822970142266*i_W2mv_a_r - 0.0305931173774017*i_W2mv_c_i - 0.0992822970142214*i_W2mv_c_r + 0.00116793362771941*i_W3lv_a_i + 0.00391345649507333*i_W3lv_a_r - 0.000583966813859693*i_W3lv_b_i - 0.00195672824753664*i_W3lv_b_r - 0.000583966813859714*i_W3lv_c_i - 0.00195672824753669*i_W3lv_c_r + 0.0293176867112763*i_W3mv_a_i + 0.0982363113431537*i_W3mv_a_r - 0.0293176867112818*i_W3mv_c_i - 0.0982363113431486*i_W3mv_c_r + 0.00697522388628822*v_GRID_a_i - 1.71338624816169e-6*v_GRID_a_r - 0.00348761194314407*v_GRID_b_i + 8.56693124078082e-7*v_GRID_b_r - 0.00348761194314416*v_GRID_c_i + 8.56693124085475e-7*v_GRID_c_r - v_W1lv_a_i
        struct[0].g[2,0] = 0.0636395371726714*i_POI_a_i - 0.0382049854269416*i_POI_a_r - 0.127279074345344*i_POI_b_i + 0.0764099708538862*i_POI_b_r + 0.0636395371726728*i_POI_c_i - 0.0382049854269445*i_POI_c_r + 0.0971901504394967*i_POImv_a_i - 0.0280418380652299*i_POImv_a_r - 0.0971901504394895*i_POImv_b_i + 0.0280418380652406*i_POImv_b_r + 0.00199839389307366*i_W1lv_a_i - 0.000634767892296777*i_W1lv_a_r - 0.0154231877861474*i_W1lv_b_i + 0.00317393578459362*i_W1lv_b_r + 0.00199839389307372*i_W1lv_c_i - 0.000634767892296846*i_W1lv_c_r + 0.100328108879395*i_W1mv_a_i - 0.0318681229122115*i_W1mv_a_r - 0.100328108879388*i_W1mv_b_i + 0.0318681229122223*i_W1mv_b_r + 0.00197756280128896*i_W2lv_a_i - 0.000609371587085066*i_W2lv_a_r - 0.00395512560257798*i_W2lv_b_i + 0.0012187431741702*i_W2lv_b_r + 0.00197756280128902*i_W2lv_c_i - 0.000609371587085134*i_W2lv_c_r + 0.0992822970142287*i_W2mv_a_i - 0.0305931173773937*i_W2mv_a_r - 0.0992822970142215*i_W2mv_b_i + 0.0305931173774048*i_W2mv_b_r + 0.00195672824753666*i_W3lv_a_i - 0.000583966813859679*i_W3lv_a_r - 0.00391345649507338*i_W3lv_b_i + 0.00116793362771943*i_W3lv_b_r + 0.00195672824753672*i_W3lv_c_i - 0.000583966813859747*i_W3lv_c_r + 0.098236311343156*i_W3mv_a_i - 0.0293176867112742*i_W3mv_a_r - 0.0982363113431489*i_W3mv_b_i + 0.029317686711285*i_W3mv_b_r - 8.5669312404275e-7*v_GRID_a_i - 0.00348761194314409*v_GRID_a_r + 1.71338624816775e-6*v_GRID_b_i + 0.00697522388628832*v_GRID_b_r - 8.56693124121928e-7*v_GRID_c_i - 0.00348761194314422*v_GRID_c_r - v_W1lv_b_r
        struct[0].g[3,0] = -0.0382049854269416*i_POI_a_i - 0.0636395371726714*i_POI_a_r + 0.0764099708538862*i_POI_b_i + 0.127279074345344*i_POI_b_r - 0.0382049854269445*i_POI_c_i - 0.0636395371726728*i_POI_c_r - 0.0280418380652299*i_POImv_a_i - 0.0971901504394967*i_POImv_a_r + 0.0280418380652406*i_POImv_b_i + 0.0971901504394895*i_POImv_b_r - 0.000634767892296777*i_W1lv_a_i - 0.00199839389307366*i_W1lv_a_r + 0.00317393578459362*i_W1lv_b_i + 0.0154231877861474*i_W1lv_b_r - 0.000634767892296846*i_W1lv_c_i - 0.00199839389307372*i_W1lv_c_r - 0.0318681229122115*i_W1mv_a_i - 0.100328108879395*i_W1mv_a_r + 0.0318681229122223*i_W1mv_b_i + 0.100328108879388*i_W1mv_b_r - 0.000609371587085066*i_W2lv_a_i - 0.00197756280128896*i_W2lv_a_r + 0.0012187431741702*i_W2lv_b_i + 0.00395512560257798*i_W2lv_b_r - 0.000609371587085134*i_W2lv_c_i - 0.00197756280128902*i_W2lv_c_r - 0.0305931173773937*i_W2mv_a_i - 0.0992822970142287*i_W2mv_a_r + 0.0305931173774048*i_W2mv_b_i + 0.0992822970142215*i_W2mv_b_r - 0.000583966813859679*i_W3lv_a_i - 0.00195672824753666*i_W3lv_a_r + 0.00116793362771943*i_W3lv_b_i + 0.00391345649507338*i_W3lv_b_r - 0.000583966813859747*i_W3lv_c_i - 0.00195672824753672*i_W3lv_c_r - 0.0293176867112742*i_W3mv_a_i - 0.098236311343156*i_W3mv_a_r + 0.029317686711285*i_W3mv_b_i + 0.0982363113431489*i_W3mv_b_r - 0.00348761194314409*v_GRID_a_i + 8.5669312404275e-7*v_GRID_a_r + 0.00697522388628832*v_GRID_b_i - 1.71338624816775e-6*v_GRID_b_r - 0.00348761194314422*v_GRID_c_i + 8.56693124121928e-7*v_GRID_c_r - v_W1lv_b_i
        struct[0].g[4,0] = 0.0636395371726713*i_POI_a_i - 0.0382049854269434*i_POI_a_r + 0.0636395371726737*i_POI_b_i - 0.0382049854269442*i_POI_b_r - 0.127279074345345*i_POI_c_i + 0.0764099708538876*i_POI_c_r + 0.0971901504394933*i_POImv_b_i - 0.0280418380652387*i_POImv_b_r - 0.0971901504394944*i_POImv_c_i + 0.0280418380652339*i_POImv_c_r + 0.00199839389307367*i_W1lv_a_i - 0.000634767892296827*i_W1lv_a_r + 0.00199839389307374*i_W1lv_b_i - 0.000634767892296831*i_W1lv_b_r - 0.0154231877861474*i_W1lv_c_i + 0.00317393578459366*i_W1lv_c_r + 0.100328108879391*i_W1mv_b_i - 0.0318681229122202*i_W1mv_b_r - 0.100328108879392*i_W1mv_c_i + 0.0318681229122156*i_W1mv_c_r + 0.00197756280128897*i_W2lv_a_i - 0.000609371587085114*i_W2lv_a_r + 0.00197756280128904*i_W2lv_b_i - 0.00060937158708512*i_W2lv_b_r - 0.00395512560257801*i_W2lv_c_i + 0.00121874317417024*i_W2lv_c_r + 0.0992822970142254*i_W2mv_b_i - 0.0305931173774027*i_W2mv_b_r - 0.0992822970142266*i_W2mv_c_i + 0.0305931173773979*i_W2mv_c_r + 0.00195672824753667*i_W3lv_a_i - 0.000583966813859728*i_W3lv_a_r + 0.00195672824753674*i_W3lv_b_i - 0.000583966813859733*i_W3lv_b_r - 0.00391345649507341*i_W3lv_c_i + 0.00116793362771946*i_W3lv_c_r + 0.0982363113431527*i_W3mv_b_i - 0.0293176867112827*i_W3mv_b_r - 0.098236311343154*i_W3mv_c_i + 0.0293176867112781*i_W3mv_c_r - 8.56693124118069e-7*v_GRID_a_i - 0.00348761194314413*v_GRID_a_r - 8.56693124090318e-7*v_GRID_b_i - 0.00348761194314425*v_GRID_b_r + 1.71338624820762e-6*v_GRID_c_i + 0.00697522388628838*v_GRID_c_r - v_W1lv_c_r
        struct[0].g[5,0] = -0.0382049854269434*i_POI_a_i - 0.0636395371726713*i_POI_a_r - 0.0382049854269442*i_POI_b_i - 0.0636395371726737*i_POI_b_r + 0.0764099708538876*i_POI_c_i + 0.127279074345345*i_POI_c_r - 0.0280418380652387*i_POImv_b_i - 0.0971901504394933*i_POImv_b_r + 0.0280418380652339*i_POImv_c_i + 0.0971901504394944*i_POImv_c_r - 0.000634767892296827*i_W1lv_a_i - 0.00199839389307367*i_W1lv_a_r - 0.000634767892296831*i_W1lv_b_i - 0.00199839389307374*i_W1lv_b_r + 0.00317393578459366*i_W1lv_c_i + 0.0154231877861474*i_W1lv_c_r - 0.0318681229122202*i_W1mv_b_i - 0.100328108879391*i_W1mv_b_r + 0.0318681229122156*i_W1mv_c_i + 0.100328108879392*i_W1mv_c_r - 0.000609371587085114*i_W2lv_a_i - 0.00197756280128897*i_W2lv_a_r - 0.00060937158708512*i_W2lv_b_i - 0.00197756280128904*i_W2lv_b_r + 0.00121874317417024*i_W2lv_c_i + 0.00395512560257801*i_W2lv_c_r - 0.0305931173774027*i_W2mv_b_i - 0.0992822970142254*i_W2mv_b_r + 0.0305931173773979*i_W2mv_c_i + 0.0992822970142266*i_W2mv_c_r - 0.000583966813859728*i_W3lv_a_i - 0.00195672824753667*i_W3lv_a_r - 0.000583966813859733*i_W3lv_b_i - 0.00195672824753674*i_W3lv_b_r + 0.00116793362771946*i_W3lv_c_i + 0.00391345649507341*i_W3lv_c_r - 0.0293176867112827*i_W3mv_b_i - 0.0982363113431527*i_W3mv_b_r + 0.0293176867112781*i_W3mv_c_i + 0.098236311343154*i_W3mv_c_r - 0.00348761194314413*v_GRID_a_i + 8.56693124118069e-7*v_GRID_a_r - 0.00348761194314425*v_GRID_b_i + 8.56693124090318e-7*v_GRID_b_r + 0.00697522388628838*v_GRID_c_i - 1.71338624820762e-6*v_GRID_c_r - v_W1lv_c_i
        struct[0].g[6,0] = -0.127279026494919*i_POI_a_i + 0.0764096462087187*i_POI_a_r + 0.0636395132474589*i_POI_b_i - 0.0382048231043588*i_POI_b_r + 0.0636395132474605*i_POI_c_i - 0.0382048231043599*i_POI_c_r - 0.0971900621093923*i_POImv_a_i + 0.0280416326518444*i_POImv_a_r + 0.0971900621093875*i_POImv_c_i - 0.0280416326518497*i_POImv_c_r - 0.00395512560257793*i_W1lv_a_i + 0.00121874317417018*i_W1lv_a_r + 0.00197756280128894*i_W1lv_b_i - 0.000609371587085081*i_W1lv_b_r + 0.00197756280128899*i_W1lv_c_i - 0.000609371587085101*i_W1lv_c_r - 0.0992822970142262*i_W1mv_a_i + 0.0305931173773962*i_W1mv_a_r + 0.0992822970142214*i_W1mv_c_i - 0.0305931173774014*i_W1mv_c_r - 0.0153815221406103*i_W2lv_a_i + 0.00312313470615651*i_W2lv_a_r + 0.00197756107030514*i_W2lv_b_i - 0.000609367353078243*i_W2lv_b_r + 0.00197756107030519*i_W2lv_c_i - 0.000609367353078263*i_W2lv_c_r - 0.099282210111273*i_W2mv_a_i + 0.030592904811745*i_W2mv_a_r + 0.0992822101112681*i_W2mv_c_i - 0.0305929048117504*i_W2mv_c_r - 0.00391345300468828*i_W3lv_a_i + 0.00116792530215105*i_W3lv_a_r + 0.00195672650234412*i_W3lv_b_i - 0.000583962651075517*i_W3lv_b_r + 0.00195672650234417*i_W3lv_c_i - 0.000583962651075537*i_W3lv_c_r - 0.0982362237268603*i_W3mv_a_i + 0.0293174777213142*i_W3mv_a_r + 0.0982362237268554*i_W3mv_c_i - 0.0293174777213195*i_W3mv_c_r + 1.70146300431068e-6*v_GRID_a_i + 0.00697521411040091*v_GRID_a_r - 8.5073150215171e-7*v_GRID_b_i - 0.00348760705520041*v_GRID_b_r - 8.50731502159103e-7*v_GRID_c_i - 0.0034876070552005*v_GRID_c_r - v_W2lv_a_r
        struct[0].g[7,0] = 0.0764096462087187*i_POI_a_i + 0.127279026494919*i_POI_a_r - 0.0382048231043588*i_POI_b_i - 0.0636395132474589*i_POI_b_r - 0.0382048231043599*i_POI_c_i - 0.0636395132474605*i_POI_c_r + 0.0280416326518444*i_POImv_a_i + 0.0971900621093923*i_POImv_a_r - 0.0280416326518497*i_POImv_c_i - 0.0971900621093875*i_POImv_c_r + 0.00121874317417018*i_W1lv_a_i + 0.00395512560257793*i_W1lv_a_r - 0.000609371587085081*i_W1lv_b_i - 0.00197756280128894*i_W1lv_b_r - 0.000609371587085101*i_W1lv_c_i - 0.00197756280128899*i_W1lv_c_r + 0.0305931173773962*i_W1mv_a_i + 0.0992822970142262*i_W1mv_a_r - 0.0305931173774014*i_W1mv_c_i - 0.0992822970142214*i_W1mv_c_r + 0.00312313470615651*i_W2lv_a_i + 0.0153815221406103*i_W2lv_a_r - 0.000609367353078243*i_W2lv_b_i - 0.00197756107030514*i_W2lv_b_r - 0.000609367353078263*i_W2lv_c_i - 0.00197756107030519*i_W2lv_c_r + 0.030592904811745*i_W2mv_a_i + 0.099282210111273*i_W2mv_a_r - 0.0305929048117504*i_W2mv_c_i - 0.0992822101112681*i_W2mv_c_r + 0.00116792530215105*i_W3lv_a_i + 0.00391345300468828*i_W3lv_a_r - 0.000583962651075517*i_W3lv_b_i - 0.00195672650234412*i_W3lv_b_r - 0.000583962651075537*i_W3lv_c_i - 0.00195672650234417*i_W3lv_c_r + 0.0293174777213142*i_W3mv_a_i + 0.0982362237268603*i_W3mv_a_r - 0.0293174777213195*i_W3mv_c_i - 0.0982362237268554*i_W3mv_c_r + 0.00697521411040091*v_GRID_a_i - 1.70146300431068e-6*v_GRID_a_r - 0.00348760705520041*v_GRID_b_i + 8.5073150215171e-7*v_GRID_b_r - 0.0034876070552005*v_GRID_c_i + 8.50731502159103e-7*v_GRID_c_r - v_W2lv_a_i
        struct[0].g[8,0] = 0.0636395132474598*i_POI_a_i - 0.0382048231043585*i_POI_a_r - 0.127279026494921*i_POI_b_i + 0.0764096462087198*i_POI_b_r + 0.0636395132474612*i_POI_c_i - 0.0382048231043613*i_POI_c_r + 0.0971900621093946*i_POImv_a_i - 0.0280416326518423*i_POImv_a_r - 0.0971900621093876*i_POImv_b_i + 0.0280416326518527*i_POImv_b_r + 0.00197756280128896*i_W1lv_a_i - 0.000609371587085065*i_W1lv_a_r - 0.00395512560257798*i_W1lv_b_i + 0.0012187431741702*i_W1lv_b_r + 0.00197756280128902*i_W1lv_c_i - 0.000609371587085133*i_W1lv_c_r + 0.0992822970142286*i_W1mv_a_i - 0.0305931173773939*i_W1mv_a_r - 0.0992822970142216*i_W1mv_b_i + 0.0305931173774045*i_W1mv_b_r + 0.00197756107030516*i_W2lv_a_i - 0.000609367353078228*i_W2lv_a_r - 0.0153815221406104*i_W2lv_b_i + 0.00312313470615652*i_W2lv_b_r + 0.00197756107030522*i_W2lv_c_i - 0.000609367353078295*i_W2lv_c_r + 0.0992822101112753*i_W2mv_a_i - 0.0305929048117428*i_W2mv_a_r - 0.0992822101112682*i_W2mv_b_i + 0.0305929048117536*i_W2mv_b_r + 0.00195672650234414*i_W3lv_a_i - 0.000583962651075503*i_W3lv_a_r - 0.00391345300468834*i_W3lv_b_i + 0.00116792530215107*i_W3lv_b_r + 0.0019567265023442*i_W3lv_c_i - 0.00058396265107557*i_W3lv_c_r + 0.0982362237268626*i_W3mv_a_i - 0.029317477721312*i_W3mv_a_r - 0.0982362237268556*i_W3mv_b_i + 0.0293174777213226*i_W3mv_b_r - 8.50731502117679e-7*v_GRID_a_i - 0.00348760705520044*v_GRID_a_r + 1.70146300431631e-6*v_GRID_b_i + 0.00697521411040101*v_GRID_b_r - 8.50731502195773e-7*v_GRID_c_i - 0.00348760705520057*v_GRID_c_r - v_W2lv_b_r
        struct[0].g[9,0] = -0.0382048231043585*i_POI_a_i - 0.0636395132474598*i_POI_a_r + 0.0764096462087198*i_POI_b_i + 0.127279026494921*i_POI_b_r - 0.0382048231043613*i_POI_c_i - 0.0636395132474612*i_POI_c_r - 0.0280416326518423*i_POImv_a_i - 0.0971900621093946*i_POImv_a_r + 0.0280416326518527*i_POImv_b_i + 0.0971900621093876*i_POImv_b_r - 0.000609371587085065*i_W1lv_a_i - 0.00197756280128896*i_W1lv_a_r + 0.0012187431741702*i_W1lv_b_i + 0.00395512560257798*i_W1lv_b_r - 0.000609371587085133*i_W1lv_c_i - 0.00197756280128902*i_W1lv_c_r - 0.0305931173773939*i_W1mv_a_i - 0.0992822970142286*i_W1mv_a_r + 0.0305931173774045*i_W1mv_b_i + 0.0992822970142216*i_W1mv_b_r - 0.000609367353078228*i_W2lv_a_i - 0.00197756107030516*i_W2lv_a_r + 0.00312313470615652*i_W2lv_b_i + 0.0153815221406104*i_W2lv_b_r - 0.000609367353078295*i_W2lv_c_i - 0.00197756107030522*i_W2lv_c_r - 0.0305929048117428*i_W2mv_a_i - 0.0992822101112753*i_W2mv_a_r + 0.0305929048117536*i_W2mv_b_i + 0.0992822101112682*i_W2mv_b_r - 0.000583962651075503*i_W3lv_a_i - 0.00195672650234414*i_W3lv_a_r + 0.00116792530215107*i_W3lv_b_i + 0.00391345300468834*i_W3lv_b_r - 0.00058396265107557*i_W3lv_c_i - 0.0019567265023442*i_W3lv_c_r - 0.029317477721312*i_W3mv_a_i - 0.0982362237268626*i_W3mv_a_r + 0.0293174777213226*i_W3mv_b_i + 0.0982362237268556*i_W3mv_b_r - 0.00348760705520044*v_GRID_a_i + 8.50731502117679e-7*v_GRID_a_r + 0.00697521411040101*v_GRID_b_i - 1.70146300431631e-6*v_GRID_b_r - 0.00348760705520057*v_GRID_c_i + 8.50731502195773e-7*v_GRID_c_r - v_W2lv_b_i
        struct[0].g[10,0] = 0.0636395132474597*i_POI_a_i - 0.0382048231043602*i_POI_a_r + 0.0636395132474621*i_POI_b_i - 0.038204823104361*i_POI_b_r - 0.127279026494922*i_POI_c_i + 0.0764096462087212*i_POI_c_r + 0.0971900621093912*i_POImv_b_i - 0.0280416326518508*i_POImv_b_r - 0.0971900621093924*i_POImv_c_i + 0.0280416326518462*i_POImv_c_r + 0.00197756280128897*i_W1lv_a_i - 0.000609371587085115*i_W1lv_a_r + 0.00197756280128904*i_W1lv_b_i - 0.000609371587085118*i_W1lv_b_r - 0.00395512560257801*i_W1lv_c_i + 0.00121874317417023*i_W1lv_c_r + 0.0992822970142252*i_W1mv_b_i - 0.0305931173774024*i_W1mv_b_r - 0.0992822970142263*i_W1mv_c_i + 0.0305931173773979*i_W1mv_c_r + 0.00197756107030517*i_W2lv_a_i - 0.000609367353078276*i_W2lv_a_r + 0.00197756107030524*i_W2lv_b_i - 0.00060936735307828*i_W2lv_b_r - 0.0153815221406104*i_W2lv_c_i + 0.00312313470615656*i_W2lv_c_r + 0.099282210111272*i_W2mv_b_i - 0.0305929048117514*i_W2mv_b_r - 0.0992822101112732*i_W2mv_c_i + 0.0305929048117468*i_W2mv_c_r + 0.00195672650234415*i_W3lv_a_i - 0.000583962651075551*i_W3lv_a_r + 0.00195672650234422*i_W3lv_b_i - 0.000583962651075555*i_W3lv_b_r - 0.00391345300468836*i_W3lv_c_i + 0.00116792530215111*i_W3lv_c_r + 0.0982362237268593*i_W3mv_b_i - 0.0293174777213204*i_W3mv_b_r - 0.0982362237268605*i_W3mv_c_i + 0.0293174777213159*i_W3mv_c_r - 8.50731502192782e-7*v_GRID_a_i - 0.00348760705520048*v_GRID_a_r - 8.50731502164163e-7*v_GRID_b_i - 0.00348760705520059*v_GRID_b_r + 1.70146300435488e-6*v_GRID_c_i + 0.00697521411040107*v_GRID_c_r - v_W2lv_c_r
        struct[0].g[11,0] = -0.0382048231043602*i_POI_a_i - 0.0636395132474597*i_POI_a_r - 0.038204823104361*i_POI_b_i - 0.0636395132474621*i_POI_b_r + 0.0764096462087212*i_POI_c_i + 0.127279026494922*i_POI_c_r - 0.0280416326518508*i_POImv_b_i - 0.0971900621093912*i_POImv_b_r + 0.0280416326518462*i_POImv_c_i + 0.0971900621093924*i_POImv_c_r - 0.000609371587085115*i_W1lv_a_i - 0.00197756280128897*i_W1lv_a_r - 0.000609371587085118*i_W1lv_b_i - 0.00197756280128904*i_W1lv_b_r + 0.00121874317417023*i_W1lv_c_i + 0.00395512560257801*i_W1lv_c_r - 0.0305931173774024*i_W1mv_b_i - 0.0992822970142252*i_W1mv_b_r + 0.0305931173773979*i_W1mv_c_i + 0.0992822970142263*i_W1mv_c_r - 0.000609367353078276*i_W2lv_a_i - 0.00197756107030517*i_W2lv_a_r - 0.00060936735307828*i_W2lv_b_i - 0.00197756107030524*i_W2lv_b_r + 0.00312313470615656*i_W2lv_c_i + 0.0153815221406104*i_W2lv_c_r - 0.0305929048117514*i_W2mv_b_i - 0.099282210111272*i_W2mv_b_r + 0.0305929048117468*i_W2mv_c_i + 0.0992822101112732*i_W2mv_c_r - 0.000583962651075551*i_W3lv_a_i - 0.00195672650234415*i_W3lv_a_r - 0.000583962651075555*i_W3lv_b_i - 0.00195672650234422*i_W3lv_b_r + 0.00116792530215111*i_W3lv_c_i + 0.00391345300468836*i_W3lv_c_r - 0.0293174777213204*i_W3mv_b_i - 0.0982362237268593*i_W3mv_b_r + 0.0293174777213159*i_W3mv_c_i + 0.0982362237268605*i_W3mv_c_r - 0.00348760705520048*v_GRID_a_i + 8.50731502192782e-7*v_GRID_a_r - 0.00348760705520059*v_GRID_b_i + 8.50731502164163e-7*v_GRID_b_r + 0.00697521411040107*v_GRID_c_i - 1.70146300435488e-6*v_GRID_c_r - v_W2lv_c_i
        struct[0].g[12,0] = -0.127278882942674*i_POI_a_i + 0.0764086722742936*i_POI_a_r + 0.0636394414713363*i_POI_b_i - 0.0382043361371463*i_POI_b_r + 0.0636394414713378*i_POI_c_i - 0.0382043361371473*i_POI_c_r - 0.0971897971186317*i_POImv_a_i + 0.0280410164125591*i_POImv_a_r + 0.0971897971186269*i_POImv_c_i - 0.0280410164125642*i_POImv_c_r - 0.00391345649507333*i_W1lv_a_i + 0.00116793362771941*i_W1lv_a_r + 0.00195672824753664*i_W1lv_b_i - 0.000583966813859694*i_W1lv_b_r + 0.00195672824753669*i_W1lv_c_i - 0.000583966813859713*i_W1lv_c_r - 0.0982363113431535*i_W1mv_a_i + 0.0293176867112763*i_W1mv_a_r + 0.0982363113431488*i_W1mv_c_i - 0.0293176867112815*i_W1mv_c_r - 0.00391345300468828*i_W2lv_a_i + 0.00116792530215105*i_W2lv_a_r + 0.00195672650234412*i_W2lv_b_i - 0.000583962651075518*i_W2lv_b_r + 0.00195672650234417*i_W2lv_c_i - 0.000583962651075537*i_W2lv_c_r - 0.0982362237268603*i_W2mv_a_i + 0.0293174777213141*i_W2mv_a_r + 0.0982362237268555*i_W2mv_c_i - 0.0293174777213194*i_W2mv_c_r - 0.0153398425335145*i_W3lv_a_i + 0.00307230032548127*i_W3lv_a_r + 0.00195672126675721*i_W3lv_b_i - 0.000583950162740626*i_W3lv_b_r + 0.00195672126675726*i_W3lv_c_i - 0.000583950162740645*i_W3lv_c_r - 0.0982359608775115*i_W3mv_a_i + 0.0293168507523132*i_W3mv_a_r + 0.0982359608775067*i_W3mv_c_i - 0.0293168507523184*i_W3mv_c_r + 1.66569333960609e-6*v_GRID_a_i + 0.00697518478272564*v_GRID_a_r - 8.32846669799631e-7*v_GRID_b_i - 0.00348759239136277*v_GRID_b_r - 8.3284666980659e-7*v_GRID_c_i - 0.00348759239136286*v_GRID_c_r - v_W3lv_a_r
        struct[0].g[13,0] = 0.0764086722742936*i_POI_a_i + 0.127278882942674*i_POI_a_r - 0.0382043361371463*i_POI_b_i - 0.0636394414713363*i_POI_b_r - 0.0382043361371473*i_POI_c_i - 0.0636394414713378*i_POI_c_r + 0.0280410164125591*i_POImv_a_i + 0.0971897971186317*i_POImv_a_r - 0.0280410164125642*i_POImv_c_i - 0.0971897971186269*i_POImv_c_r + 0.00116793362771941*i_W1lv_a_i + 0.00391345649507333*i_W1lv_a_r - 0.000583966813859694*i_W1lv_b_i - 0.00195672824753664*i_W1lv_b_r - 0.000583966813859713*i_W1lv_c_i - 0.00195672824753669*i_W1lv_c_r + 0.0293176867112763*i_W1mv_a_i + 0.0982363113431535*i_W1mv_a_r - 0.0293176867112815*i_W1mv_c_i - 0.0982363113431488*i_W1mv_c_r + 0.00116792530215105*i_W2lv_a_i + 0.00391345300468828*i_W2lv_a_r - 0.000583962651075518*i_W2lv_b_i - 0.00195672650234412*i_W2lv_b_r - 0.000583962651075537*i_W2lv_c_i - 0.00195672650234417*i_W2lv_c_r + 0.0293174777213141*i_W2mv_a_i + 0.0982362237268603*i_W2mv_a_r - 0.0293174777213194*i_W2mv_c_i - 0.0982362237268555*i_W2mv_c_r + 0.00307230032548127*i_W3lv_a_i + 0.0153398425335145*i_W3lv_a_r - 0.000583950162740626*i_W3lv_b_i - 0.00195672126675721*i_W3lv_b_r - 0.000583950162740645*i_W3lv_c_i - 0.00195672126675726*i_W3lv_c_r + 0.0293168507523132*i_W3mv_a_i + 0.0982359608775115*i_W3mv_a_r - 0.0293168507523184*i_W3mv_c_i - 0.0982359608775067*i_W3mv_c_r + 0.00697518478272564*v_GRID_a_i - 1.66569333960609e-6*v_GRID_a_r - 0.00348759239136277*v_GRID_b_i + 8.32846669799631e-7*v_GRID_b_r - 0.00348759239136286*v_GRID_c_i + 8.3284666980659e-7*v_GRID_c_r - v_W3lv_a_i
        struct[0].g[14,0] = 0.0636394414713372*i_POI_a_i - 0.0382043361371459*i_POI_a_r - 0.127278882942676*i_POI_b_i + 0.0764086722742947*i_POI_b_r + 0.0636394414713386*i_POI_c_i - 0.0382043361371487*i_POI_c_r + 0.097189797118634*i_POImv_a_i - 0.028041016412557*i_POImv_a_r - 0.0971897971186271*i_POImv_b_i + 0.0280410164125671*i_POImv_b_r + 0.00195672824753666*i_W1lv_a_i - 0.00058396681385968*i_W1lv_a_r - 0.00391345649507338*i_W1lv_b_i + 0.00116793362771943*i_W1lv_b_r + 0.00195672824753672*i_W1lv_c_i - 0.000583966813859745*i_W1lv_c_r + 0.0982363113431559*i_W1mv_a_i - 0.0293176867112742*i_W1mv_a_r - 0.098236311343149*i_W1mv_b_i + 0.0293176867112845*i_W1mv_b_r + 0.00195672650234414*i_W2lv_a_i - 0.000583962651075503*i_W2lv_a_r - 0.00391345300468834*i_W2lv_b_i + 0.00116792530215107*i_W2lv_b_r + 0.0019567265023442*i_W2lv_c_i - 0.000583962651075569*i_W2lv_c_r + 0.0982362237268626*i_W2mv_a_i - 0.029317477721312*i_W2mv_a_r - 0.0982362237268557*i_W2mv_b_i + 0.0293174777213224*i_W2mv_b_r + 0.00195672126675723*i_W3lv_a_i - 0.000583950162740612*i_W3lv_a_r - 0.0153398425335145*i_W3lv_b_i + 0.00307230032548129*i_W3lv_b_r + 0.00195672126675729*i_W3lv_c_i - 0.000583950162740677*i_W3lv_c_r + 0.0982359608775139*i_W3mv_a_i - 0.0293168507523111*i_W3mv_a_r - 0.0982359608775069*i_W3mv_b_i + 0.0293168507523214*i_W3mv_b_r - 8.32846669765817e-7*v_GRID_a_i - 0.0034875923913628*v_GRID_a_r + 1.66569333961041e-6*v_GRID_b_i + 0.00697518478272573*v_GRID_b_r - 8.32846669843044e-7*v_GRID_c_i - 0.00348759239136293*v_GRID_c_r - v_W3lv_b_r
        struct[0].g[15,0] = -0.0382043361371459*i_POI_a_i - 0.0636394414713372*i_POI_a_r + 0.0764086722742947*i_POI_b_i + 0.127278882942676*i_POI_b_r - 0.0382043361371487*i_POI_c_i - 0.0636394414713386*i_POI_c_r - 0.028041016412557*i_POImv_a_i - 0.097189797118634*i_POImv_a_r + 0.0280410164125671*i_POImv_b_i + 0.0971897971186271*i_POImv_b_r - 0.00058396681385968*i_W1lv_a_i - 0.00195672824753666*i_W1lv_a_r + 0.00116793362771943*i_W1lv_b_i + 0.00391345649507338*i_W1lv_b_r - 0.000583966813859745*i_W1lv_c_i - 0.00195672824753672*i_W1lv_c_r - 0.0293176867112742*i_W1mv_a_i - 0.0982363113431559*i_W1mv_a_r + 0.0293176867112845*i_W1mv_b_i + 0.098236311343149*i_W1mv_b_r - 0.000583962651075503*i_W2lv_a_i - 0.00195672650234414*i_W2lv_a_r + 0.00116792530215107*i_W2lv_b_i + 0.00391345300468834*i_W2lv_b_r - 0.000583962651075569*i_W2lv_c_i - 0.0019567265023442*i_W2lv_c_r - 0.029317477721312*i_W2mv_a_i - 0.0982362237268626*i_W2mv_a_r + 0.0293174777213224*i_W2mv_b_i + 0.0982362237268557*i_W2mv_b_r - 0.000583950162740612*i_W3lv_a_i - 0.00195672126675723*i_W3lv_a_r + 0.00307230032548129*i_W3lv_b_i + 0.0153398425335145*i_W3lv_b_r - 0.000583950162740677*i_W3lv_c_i - 0.00195672126675729*i_W3lv_c_r - 0.0293168507523111*i_W3mv_a_i - 0.0982359608775139*i_W3mv_a_r + 0.0293168507523214*i_W3mv_b_i + 0.0982359608775069*i_W3mv_b_r - 0.0034875923913628*v_GRID_a_i + 8.32846669765817e-7*v_GRID_a_r + 0.00697518478272573*v_GRID_b_i - 1.66569333961041e-6*v_GRID_b_r - 0.00348759239136293*v_GRID_c_i + 8.32846669843044e-7*v_GRID_c_r - v_W3lv_b_i
        struct[0].g[16,0] = 0.063639441471337*i_POI_a_i - 0.0382043361371477*i_POI_a_r + 0.0636394414713394*i_POI_b_i - 0.0382043361371484*i_POI_b_r - 0.127278882942676*i_POI_c_i + 0.0764086722742961*i_POI_c_r + 0.0971897971186306*i_POImv_b_i - 0.0280410164125652*i_POImv_b_r - 0.0971897971186319*i_POImv_c_i + 0.0280410164125607*i_POImv_c_r + 0.00195672824753667*i_W1lv_a_i - 0.000583966813859728*i_W1lv_a_r + 0.00195672824753674*i_W1lv_b_i - 0.000583966813859731*i_W1lv_b_r - 0.00391345649507341*i_W1lv_c_i + 0.00116793362771946*i_W1lv_c_r + 0.0982363113431526*i_W1mv_b_i - 0.0293176867112824*i_W1mv_b_r - 0.0982363113431537*i_W1mv_c_i + 0.0293176867112781*i_W1mv_c_r + 0.00195672650234415*i_W2lv_a_i - 0.000583962651075551*i_W2lv_a_r + 0.00195672650234422*i_W2lv_b_i - 0.000583962651075554*i_W2lv_b_r - 0.00391345300468836*i_W2lv_c_i + 0.00116792530215111*i_W2lv_c_r + 0.0982362237268593*i_W2mv_b_i - 0.0293174777213203*i_W2mv_b_r - 0.0982362237268605*i_W2mv_c_i + 0.0293174777213159*i_W2mv_c_r + 0.00195672126675724*i_W3lv_a_i - 0.000583950162740659*i_W3lv_a_r + 0.00195672126675731*i_W3lv_b_i - 0.000583950162740663*i_W3lv_b_r - 0.0153398425335145*i_W3lv_c_i + 0.00307230032548132*i_W3lv_c_r + 0.0982359608775105*i_W3mv_b_i - 0.0293168507523193*i_W3mv_b_r - 0.0982359608775118*i_W3mv_c_i + 0.0293168507523149*i_W3mv_c_r - 8.32846669840052e-7*v_GRID_a_i - 0.00348759239136284*v_GRID_a_r - 8.32846669810349e-7*v_GRID_b_i - 0.00348759239136295*v_GRID_b_r + 1.66569333964898e-6*v_GRID_c_i + 0.00697518478272579*v_GRID_c_r - v_W3lv_c_r
        struct[0].g[17,0] = -0.0382043361371477*i_POI_a_i - 0.063639441471337*i_POI_a_r - 0.0382043361371484*i_POI_b_i - 0.0636394414713394*i_POI_b_r + 0.0764086722742961*i_POI_c_i + 0.127278882942676*i_POI_c_r - 0.0280410164125652*i_POImv_b_i - 0.0971897971186306*i_POImv_b_r + 0.0280410164125607*i_POImv_c_i + 0.0971897971186319*i_POImv_c_r - 0.000583966813859728*i_W1lv_a_i - 0.00195672824753667*i_W1lv_a_r - 0.000583966813859731*i_W1lv_b_i - 0.00195672824753674*i_W1lv_b_r + 0.00116793362771946*i_W1lv_c_i + 0.00391345649507341*i_W1lv_c_r - 0.0293176867112824*i_W1mv_b_i - 0.0982363113431526*i_W1mv_b_r + 0.0293176867112781*i_W1mv_c_i + 0.0982363113431537*i_W1mv_c_r - 0.000583962651075551*i_W2lv_a_i - 0.00195672650234415*i_W2lv_a_r - 0.000583962651075554*i_W2lv_b_i - 0.00195672650234422*i_W2lv_b_r + 0.00116792530215111*i_W2lv_c_i + 0.00391345300468836*i_W2lv_c_r - 0.0293174777213203*i_W2mv_b_i - 0.0982362237268593*i_W2mv_b_r + 0.0293174777213159*i_W2mv_c_i + 0.0982362237268605*i_W2mv_c_r - 0.000583950162740659*i_W3lv_a_i - 0.00195672126675724*i_W3lv_a_r - 0.000583950162740663*i_W3lv_b_i - 0.00195672126675731*i_W3lv_b_r + 0.00307230032548132*i_W3lv_c_i + 0.0153398425335145*i_W3lv_c_r - 0.0293168507523193*i_W3mv_b_i - 0.0982359608775105*i_W3mv_b_r + 0.0293168507523149*i_W3mv_c_i + 0.0982359608775118*i_W3mv_c_r - 0.00348759239136284*v_GRID_a_i + 8.32846669840052e-7*v_GRID_a_r - 0.00348759239136295*v_GRID_b_i + 8.32846669810349e-7*v_GRID_b_r + 0.00697518478272579*v_GRID_c_i - 1.66569333964898e-6*v_GRID_c_r - v_W3lv_c_i
        struct[0].g[18,0] = -3.19497213887022*i_POI_a_i + 1.9179839277919*i_POI_a_r + 3.19497213887041*i_POI_b_i - 1.9179839277919*i_POI_b_r - 3.08885814101954*i_POImv_a_i + 11.6022742434667*i_POImv_a_r + 1.79047234076949*i_POImv_b_i + 10.1945442087447*i_POImv_b_r + 1.79047234076923*i_POImv_c_i + 10.1945442087447*i_POImv_c_r - 0.0971901504394881*i_W1lv_a_i + 0.028041838065235*i_W1lv_a_r + 0.0971901504394933*i_W1lv_b_i - 0.0280418380652335*i_W1lv_b_r - 3.08738963687028*i_W1mv_a_i + 11.6035242966407*i_W1mv_a_r + 1.79198075607073*i_W1mv_b_i + 10.1957014483333*i_W1mv_b_r + 1.79198075607047*i_W1mv_c_i + 10.1957014483332*i_W1mv_c_r - 0.0971900621093861*i_W2lv_a_i + 0.0280416326518473*i_W2lv_a_r + 0.0971900621093912*i_W2lv_b_i - 0.0280416326518457*i_W2lv_b_r - 3.08755280951041*i_W2mv_a_i + 11.60338540301*i_W2mv_a_r + 1.79181314887337*i_W2mv_b_i + 10.1955728673526*i_W2mv_b_r + 1.79181314887311*i_W2mv_c_i + 10.1955728673525*i_W2mv_c_r - 0.0971897971186256*i_W3lv_a_i + 0.028041016412562*i_W3lv_a_r + 0.0971897971186307*i_W3lv_b_i - 0.0280410164125604*i_W3lv_b_r - 3.08804231916211*i_W3mv_a_i + 11.6029687203683*i_W3mv_a_r + 1.79131033552716*i_W3mv_b_i + 10.1951871226168*i_W3mv_b_r + 1.7913103355269*i_W3mv_c_i + 10.1951871226167*i_W3mv_c_r + 4.03160543824476e-5*v_GRID_a_i + 0.175091156146066*v_GRID_a_r - 4.03160543774783e-5*v_GRID_b_i - 0.175091156146074*v_GRID_b_r - 3.9510886540828e-17*v_GRID_c_i + 3.80491223180095e-17*v_GRID_c_r - v_POImv_a_r
        struct[0].g[19,0] = 1.9179839277919*i_POI_a_i + 3.19497213887022*i_POI_a_r - 1.9179839277919*i_POI_b_i - 3.19497213887041*i_POI_b_r + 11.6022742434667*i_POImv_a_i + 3.08885814101954*i_POImv_a_r + 10.1945442087447*i_POImv_b_i - 1.79047234076949*i_POImv_b_r + 10.1945442087447*i_POImv_c_i - 1.79047234076923*i_POImv_c_r + 0.028041838065235*i_W1lv_a_i + 0.0971901504394881*i_W1lv_a_r - 0.0280418380652335*i_W1lv_b_i - 0.0971901504394933*i_W1lv_b_r + 11.6035242966407*i_W1mv_a_i + 3.08738963687028*i_W1mv_a_r + 10.1957014483333*i_W1mv_b_i - 1.79198075607073*i_W1mv_b_r + 10.1957014483332*i_W1mv_c_i - 1.79198075607047*i_W1mv_c_r + 0.0280416326518473*i_W2lv_a_i + 0.0971900621093861*i_W2lv_a_r - 0.0280416326518457*i_W2lv_b_i - 0.0971900621093912*i_W2lv_b_r + 11.60338540301*i_W2mv_a_i + 3.08755280951041*i_W2mv_a_r + 10.1955728673526*i_W2mv_b_i - 1.79181314887337*i_W2mv_b_r + 10.1955728673525*i_W2mv_c_i - 1.79181314887311*i_W2mv_c_r + 0.028041016412562*i_W3lv_a_i + 0.0971897971186256*i_W3lv_a_r - 0.0280410164125604*i_W3lv_b_i - 0.0971897971186307*i_W3lv_b_r + 11.6029687203683*i_W3mv_a_i + 3.08804231916211*i_W3mv_a_r + 10.1951871226168*i_W3mv_b_i - 1.79131033552716*i_W3mv_b_r + 10.1951871226167*i_W3mv_c_i - 1.7913103355269*i_W3mv_c_r + 0.175091156146066*v_GRID_a_i - 4.03160543824476e-5*v_GRID_a_r - 0.175091156146074*v_GRID_b_i + 4.03160543774783e-5*v_GRID_b_r + 3.80491223180095e-17*v_GRID_c_i + 3.9510886540828e-17*v_GRID_c_r - v_POImv_a_i
        struct[0].g[20,0] = -3.19497213887037*i_POI_b_i + 1.91798392779201*i_POI_b_r + 3.19497213887023*i_POI_c_i - 1.91798392779203*i_POI_c_r + 1.79047234076965*i_POImv_a_i + 10.1945442087449*i_POImv_a_r - 3.08885814101935*i_POImv_b_i + 11.6022742434671*i_POImv_b_r + 1.79047234076948*i_POImv_c_i + 10.1945442087448*i_POImv_c_r - 0.097190150439493*i_W1lv_b_i + 0.028041838065237*i_W1lv_b_r + 0.0971901504394895*i_W1lv_c_i - 0.0280418380652384*i_W1lv_c_r + 1.7919807560709*i_W1mv_a_i + 10.1957014483335*i_W1mv_a_r - 3.0873896368701*i_W1mv_b_i + 11.603524296641*i_W1mv_b_r + 1.79198075607072*i_W1mv_c_i + 10.1957014483334*i_W1mv_c_r - 0.097190062109391*i_W2lv_b_i + 0.0280416326518492*i_W2lv_b_r + 0.0971900621093875*i_W2lv_c_i - 0.0280416326518506*i_W2lv_c_r + 1.79181314887354*i_W2mv_a_i + 10.1955728673528*i_W2mv_a_r - 3.08755280951023*i_W2mv_b_i + 11.6033854030103*i_W2mv_b_r + 1.79181314887336*i_W2mv_c_i + 10.1955728673527*i_W2mv_c_r - 0.0971897971186304*i_W3lv_b_i + 0.0280410164125637*i_W3lv_b_r + 0.0971897971186269*i_W3lv_c_i - 0.0280410164125652*i_W3lv_c_r + 1.79131033552732*i_W3mv_a_i + 10.1951871226169*i_W3mv_a_r - 3.08804231916193*i_W3mv_b_i + 11.6029687203686*i_W3mv_b_r + 1.79131033552715*i_W3mv_c_i + 10.1951871226169*i_W3mv_c_r + 4.89567608905926e-18*v_GRID_a_i + 2.93587278316694e-18*v_GRID_a_r + 4.03160543830677e-5*v_GRID_b_i + 0.175091156146075*v_GRID_b_r - 4.03160543870874e-5*v_GRID_c_i - 0.17509115614607*v_GRID_c_r - v_POImv_b_r
        struct[0].g[21,0] = 1.91798392779201*i_POI_b_i + 3.19497213887037*i_POI_b_r - 1.91798392779203*i_POI_c_i - 3.19497213887023*i_POI_c_r + 10.1945442087449*i_POImv_a_i - 1.79047234076965*i_POImv_a_r + 11.6022742434671*i_POImv_b_i + 3.08885814101935*i_POImv_b_r + 10.1945442087448*i_POImv_c_i - 1.79047234076948*i_POImv_c_r + 0.028041838065237*i_W1lv_b_i + 0.097190150439493*i_W1lv_b_r - 0.0280418380652384*i_W1lv_c_i - 0.0971901504394895*i_W1lv_c_r + 10.1957014483335*i_W1mv_a_i - 1.7919807560709*i_W1mv_a_r + 11.603524296641*i_W1mv_b_i + 3.0873896368701*i_W1mv_b_r + 10.1957014483334*i_W1mv_c_i - 1.79198075607072*i_W1mv_c_r + 0.0280416326518492*i_W2lv_b_i + 0.097190062109391*i_W2lv_b_r - 0.0280416326518506*i_W2lv_c_i - 0.0971900621093875*i_W2lv_c_r + 10.1955728673528*i_W2mv_a_i - 1.79181314887354*i_W2mv_a_r + 11.6033854030103*i_W2mv_b_i + 3.08755280951023*i_W2mv_b_r + 10.1955728673527*i_W2mv_c_i - 1.79181314887336*i_W2mv_c_r + 0.0280410164125637*i_W3lv_b_i + 0.0971897971186304*i_W3lv_b_r - 0.0280410164125652*i_W3lv_c_i - 0.0971897971186269*i_W3lv_c_r + 10.1951871226169*i_W3mv_a_i - 1.79131033552732*i_W3mv_a_r + 11.6029687203686*i_W3mv_b_i + 3.08804231916193*i_W3mv_b_r + 10.1951871226169*i_W3mv_c_i - 1.79131033552715*i_W3mv_c_r + 2.93587278316694e-18*v_GRID_a_i - 4.89567608905926e-18*v_GRID_a_r + 0.175091156146075*v_GRID_b_i - 4.03160543830677e-5*v_GRID_b_r - 0.17509115614607*v_GRID_c_i + 4.03160543870874e-5*v_GRID_c_r - v_POImv_b_i
        struct[0].g[22,0] = 3.19497213887049*i_POI_a_i - 1.91798392779195*i_POI_a_r - 3.19497213887058*i_POI_c_i + 1.91798392779195*i_POI_c_r + 1.79047234076953*i_POImv_a_i + 10.1945442087448*i_POImv_a_r + 1.79047234076966*i_POImv_b_i + 10.1945442087448*i_POImv_b_r - 3.0888581410196*i_POImv_c_i + 11.6022742434669*i_POImv_c_r + 0.0971901504394956*i_W1lv_a_i - 0.0280418380652345*i_W1lv_a_r - 0.0971901504394982*i_W1lv_c_i + 0.0280418380652337*i_W1lv_c_r + 1.79198075607078*i_W1mv_a_i + 10.1957014483334*i_W1mv_a_r + 1.79198075607091*i_W1mv_b_i + 10.1957014483334*i_W1mv_b_r - 3.08738963687035*i_W1mv_c_i + 11.6035242966408*i_W1mv_c_r + 0.0971900621093936*i_W2lv_a_i - 0.0280416326518468*i_W2lv_a_r - 0.0971900621093962*i_W2lv_c_i + 0.028041632651846*i_W2lv_c_r + 1.79181314887342*i_W2mv_a_i + 10.1955728673527*i_W2mv_a_r + 1.79181314887355*i_W2mv_b_i + 10.1955728673527*i_W2mv_b_r - 3.08755280951048*i_W2mv_c_i + 11.6033854030101*i_W2mv_c_r + 0.0971897971186331*i_W3lv_a_i - 0.0280410164125613*i_W3lv_a_r - 0.0971897971186357*i_W3lv_c_i + 0.0280410164125606*i_W3lv_c_r + 1.79131033552721*i_W3mv_a_i + 10.1951871226168*i_W3mv_a_r + 1.79131033552734*i_W3mv_b_i + 10.1951871226169*i_W3mv_b_r - 3.08804231916218*i_W3mv_c_i + 11.6029687203684*i_W3mv_c_r - 4.03160543779422e-5*v_GRID_a_i - 0.175091156146078*v_GRID_a_r + 3.10999136728748e-17*v_GRID_b_i - 4.08928123428111e-17*v_GRID_b_r + 4.03160543753863e-5*v_GRID_c_i + 0.175091156146082*v_GRID_c_r - v_POImv_c_r
        struct[0].g[23,0] = -1.91798392779195*i_POI_a_i - 3.19497213887049*i_POI_a_r + 1.91798392779195*i_POI_c_i + 3.19497213887058*i_POI_c_r + 10.1945442087448*i_POImv_a_i - 1.79047234076953*i_POImv_a_r + 10.1945442087448*i_POImv_b_i - 1.79047234076966*i_POImv_b_r + 11.6022742434669*i_POImv_c_i + 3.0888581410196*i_POImv_c_r - 0.0280418380652345*i_W1lv_a_i - 0.0971901504394956*i_W1lv_a_r + 0.0280418380652337*i_W1lv_c_i + 0.0971901504394982*i_W1lv_c_r + 10.1957014483334*i_W1mv_a_i - 1.79198075607078*i_W1mv_a_r + 10.1957014483334*i_W1mv_b_i - 1.79198075607091*i_W1mv_b_r + 11.6035242966408*i_W1mv_c_i + 3.08738963687035*i_W1mv_c_r - 0.0280416326518468*i_W2lv_a_i - 0.0971900621093936*i_W2lv_a_r + 0.028041632651846*i_W2lv_c_i + 0.0971900621093962*i_W2lv_c_r + 10.1955728673527*i_W2mv_a_i - 1.79181314887342*i_W2mv_a_r + 10.1955728673527*i_W2mv_b_i - 1.79181314887355*i_W2mv_b_r + 11.6033854030101*i_W2mv_c_i + 3.08755280951048*i_W2mv_c_r - 0.0280410164125613*i_W3lv_a_i - 0.0971897971186331*i_W3lv_a_r + 0.0280410164125606*i_W3lv_c_i + 0.0971897971186357*i_W3lv_c_r + 10.1951871226168*i_W3mv_a_i - 1.79131033552721*i_W3mv_a_r + 10.1951871226169*i_W3mv_b_i - 1.79131033552734*i_W3mv_b_r + 11.6029687203684*i_W3mv_c_i + 3.08804231916218*i_W3mv_c_r - 0.175091156146078*v_GRID_a_i + 4.03160543779422e-5*v_GRID_a_r - 4.08928123428111e-17*v_GRID_b_i - 3.10999136728748e-17*v_GRID_b_r + 0.175091156146082*v_GRID_c_i - 4.03160543753863e-5*v_GRID_c_r - v_POImv_c_i
        struct[0].g[24,0] = -16.3488065237228*i_POI_a_i + 8.99352976113163*i_POI_a_r + 1.90429395893588*i_POI_b_i - 1.96237550602395*i_POI_b_r + 1.90429395893593*i_POI_c_i - 1.962375506024*i_POI_c_r - 3.19497213887046*i_POImv_a_i + 1.91798392779186*i_POImv_a_r + 3.19497213887024*i_POImv_c_i - 1.91798392779199*i_POImv_c_r - 0.127279074345343*i_W1lv_a_i + 0.076409970853885*i_W1lv_a_r + 0.0636395371726706*i_W1lv_b_i - 0.0382049854269419*i_W1lv_b_r + 0.0636395371726721*i_W1lv_c_i - 0.038204985426943*i_W1lv_c_r - 3.19498294936924*i_W1mv_a_i + 1.91805727135914*i_W1mv_a_r + 3.19498294936902*i_W1mv_c_i - 1.91805727135928*i_W1mv_c_r - 0.127279026494919*i_W2lv_a_i + 0.0764096462087186*i_W2lv_a_r + 0.063639513247459*i_W2lv_b_i - 0.0382048231043588*i_W2lv_b_r + 0.0636395132474605*i_W2lv_c_i - 0.0382048231043599*i_W2lv_c_r - 3.19498174821903*i_W2mv_a_i + 1.91804912205592*i_W2mv_a_r + 3.19498174821881*i_W2mv_c_i - 1.91804912205606*i_W2mv_c_r - 0.127278882942674*i_W3lv_a_i + 0.0764086722742935*i_W3lv_a_r + 0.0636394414713363*i_W3lv_b_i - 0.0382043361371462*i_W3lv_b_r + 0.0636394414713379*i_W3lv_c_i - 0.0382043361371473*i_W3lv_c_r - 3.19497814474392*i_W3mv_a_i + 1.9180246741732*i_W3mv_a_r + 3.19497814474371*i_W3mv_c_i - 1.91802467417334*i_W3mv_c_r - 0.0328668071354569*v_GRID_a_i + 0.876104930717234*v_GRID_a_r - 0.0330297796399041*v_GRID_b_i - 0.124162742246183*v_GRID_b_r - 0.0330297796399051*v_GRID_c_i - 0.124162742246186*v_GRID_c_r - v_POI_a_r
        struct[0].g[25,0] = 8.99352976113163*i_POI_a_i + 16.3488065237228*i_POI_a_r - 1.96237550602395*i_POI_b_i - 1.90429395893588*i_POI_b_r - 1.962375506024*i_POI_c_i - 1.90429395893593*i_POI_c_r + 1.91798392779186*i_POImv_a_i + 3.19497213887046*i_POImv_a_r - 1.91798392779199*i_POImv_c_i - 3.19497213887024*i_POImv_c_r + 0.076409970853885*i_W1lv_a_i + 0.127279074345343*i_W1lv_a_r - 0.0382049854269419*i_W1lv_b_i - 0.0636395371726706*i_W1lv_b_r - 0.038204985426943*i_W1lv_c_i - 0.0636395371726721*i_W1lv_c_r + 1.91805727135914*i_W1mv_a_i + 3.19498294936924*i_W1mv_a_r - 1.91805727135928*i_W1mv_c_i - 3.19498294936902*i_W1mv_c_r + 0.0764096462087186*i_W2lv_a_i + 0.127279026494919*i_W2lv_a_r - 0.0382048231043588*i_W2lv_b_i - 0.063639513247459*i_W2lv_b_r - 0.0382048231043599*i_W2lv_c_i - 0.0636395132474605*i_W2lv_c_r + 1.91804912205592*i_W2mv_a_i + 3.19498174821903*i_W2mv_a_r - 1.91804912205606*i_W2mv_c_i - 3.19498174821881*i_W2mv_c_r + 0.0764086722742935*i_W3lv_a_i + 0.127278882942674*i_W3lv_a_r - 0.0382043361371462*i_W3lv_b_i - 0.0636394414713363*i_W3lv_b_r - 0.0382043361371473*i_W3lv_c_i - 0.0636394414713379*i_W3lv_c_r + 1.9180246741732*i_W3mv_a_i + 3.19497814474392*i_W3mv_a_r - 1.91802467417334*i_W3mv_c_i - 3.19497814474371*i_W3mv_c_r + 0.876104930717234*v_GRID_a_i + 0.0328668071354569*v_GRID_a_r - 0.124162742246183*v_GRID_b_i + 0.0330297796399041*v_GRID_b_r - 0.124162742246186*v_GRID_c_i + 0.0330297796399051*v_GRID_c_r - v_POI_a_i
        struct[0].g[26,0] = 1.90429395893591*i_POI_a_i - 1.96237550602395*i_POI_a_r - 16.3488065237228*i_POI_b_i + 8.99352976113168*i_POI_b_r + 1.90429395893593*i_POI_c_i - 1.96237550602406*i_POI_c_r + 3.19497213887056*i_POImv_a_i - 1.91798392779181*i_POImv_a_r - 3.19497213887022*i_POImv_b_i + 1.9179839277921*i_POImv_b_r + 0.0636395371726714*i_W1lv_a_i - 0.0382049854269416*i_W1lv_a_r - 0.127279074345344*i_W1lv_b_i + 0.0764099708538861*i_W1lv_b_r + 0.0636395371726729*i_W1lv_c_i - 0.0382049854269445*i_W1lv_c_r + 3.19498294936934*i_W1mv_a_i - 1.91805727135909*i_W1mv_a_r - 3.194982949369*i_W1mv_b_i + 1.91805727135939*i_W1mv_b_r + 0.0636395132474598*i_W2lv_a_i - 0.0382048231043584*i_W2lv_a_r - 0.127279026494921*i_W2lv_b_i + 0.0764096462087197*i_W2lv_b_r + 0.0636395132474612*i_W2lv_c_i - 0.0382048231043613*i_W2lv_c_r + 3.19498174821913*i_W2mv_a_i - 1.91804912205587*i_W2mv_a_r - 3.19498174821879*i_W2mv_b_i + 1.91804912205617*i_W2mv_b_r + 0.0636394414713371*i_W3lv_a_i - 0.0382043361371459*i_W3lv_a_r - 0.127278882942676*i_W3lv_b_i + 0.0764086722742946*i_W3lv_b_r + 0.0636394414713386*i_W3lv_c_i - 0.0382043361371488*i_W3lv_c_r + 3.19497814474403*i_W3mv_a_i - 1.91802467417315*i_W3mv_a_r - 3.19497814474369*i_W3mv_b_i + 1.91802467417345*i_W3mv_b_r - 0.0330297796399032*v_GRID_a_i - 0.124162742246184*v_GRID_a_r - 0.0328668071354559*v_GRID_b_i + 0.876104930717237*v_GRID_b_r - 0.033029779639907*v_GRID_c_i - 0.124162742246188*v_GRID_c_r - v_POI_b_r
        struct[0].g[27,0] = -1.96237550602395*i_POI_a_i - 1.90429395893591*i_POI_a_r + 8.99352976113168*i_POI_b_i + 16.3488065237228*i_POI_b_r - 1.96237550602406*i_POI_c_i - 1.90429395893593*i_POI_c_r - 1.91798392779181*i_POImv_a_i - 3.19497213887056*i_POImv_a_r + 1.9179839277921*i_POImv_b_i + 3.19497213887022*i_POImv_b_r - 0.0382049854269416*i_W1lv_a_i - 0.0636395371726714*i_W1lv_a_r + 0.0764099708538861*i_W1lv_b_i + 0.127279074345344*i_W1lv_b_r - 0.0382049854269445*i_W1lv_c_i - 0.0636395371726729*i_W1lv_c_r - 1.91805727135909*i_W1mv_a_i - 3.19498294936934*i_W1mv_a_r + 1.91805727135939*i_W1mv_b_i + 3.194982949369*i_W1mv_b_r - 0.0382048231043584*i_W2lv_a_i - 0.0636395132474598*i_W2lv_a_r + 0.0764096462087197*i_W2lv_b_i + 0.127279026494921*i_W2lv_b_r - 0.0382048231043613*i_W2lv_c_i - 0.0636395132474612*i_W2lv_c_r - 1.91804912205587*i_W2mv_a_i - 3.19498174821913*i_W2mv_a_r + 1.91804912205617*i_W2mv_b_i + 3.19498174821879*i_W2mv_b_r - 0.0382043361371459*i_W3lv_a_i - 0.0636394414713371*i_W3lv_a_r + 0.0764086722742946*i_W3lv_b_i + 0.127278882942676*i_W3lv_b_r - 0.0382043361371488*i_W3lv_c_i - 0.0636394414713386*i_W3lv_c_r - 1.91802467417315*i_W3mv_a_i - 3.19497814474403*i_W3mv_a_r + 1.91802467417345*i_W3mv_b_i + 3.19497814474369*i_W3mv_b_r - 0.124162742246184*v_GRID_a_i + 0.0330297796399032*v_GRID_a_r + 0.876104930717237*v_GRID_b_i + 0.0328668071354559*v_GRID_b_r - 0.124162742246188*v_GRID_c_i + 0.033029779639907*v_GRID_c_r - v_POI_b_i
        struct[0].g[28,0] = 1.90429395893589*i_POI_a_i - 1.96237550602401*i_POI_a_r + 1.90429395893597*i_POI_b_i - 1.96237550602406*i_POI_b_r - 16.3488065237228*i_POI_c_i + 8.99352976113174*i_POI_c_r + 3.19497213887036*i_POImv_b_i - 1.91798392779206*i_POImv_b_r - 3.19497213887045*i_POImv_c_i + 1.91798392779192*i_POImv_c_r + 0.0636395371726713*i_W1lv_a_i - 0.0382049854269434*i_W1lv_a_r + 0.0636395371726737*i_W1lv_b_i - 0.0382049854269442*i_W1lv_b_r - 0.127279074345345*i_W1lv_c_i + 0.0764099708538875*i_W1lv_c_r + 3.19498294936915*i_W1mv_b_i - 1.91805727135935*i_W1mv_b_r - 3.19498294936923*i_W1mv_c_i + 1.91805727135921*i_W1mv_c_r + 0.0636395132474596*i_W2lv_a_i - 0.0382048231043602*i_W2lv_a_r + 0.0636395132474621*i_W2lv_b_i - 0.038204823104361*i_W2lv_b_r - 0.127279026494922*i_W2lv_c_i + 0.0764096462087212*i_W2lv_c_r + 3.19498174821894*i_W2mv_b_i - 1.91804912205613*i_W2mv_b_r - 3.19498174821902*i_W2mv_c_i + 1.91804912205598*i_W2mv_c_r + 0.063639441471337*i_W3lv_a_i - 0.0382043361371476*i_W3lv_a_r + 0.0636394414713394*i_W3lv_b_i - 0.0382043361371484*i_W3lv_b_r - 0.127278882942676*i_W3lv_c_i + 0.0764086722742961*i_W3lv_c_r + 3.19497814474383*i_W3mv_b_i - 1.91802467417341*i_W3mv_b_r - 3.19497814474392*i_W3mv_c_i + 1.91802467417326*i_W3mv_c_r - 0.0330297796399061*v_GRID_a_i - 0.124162742246184*v_GRID_a_r - 0.0330297796399062*v_GRID_b_i - 0.124162742246189*v_GRID_b_r - 0.032866807135454*v_GRID_c_i + 0.876104930717239*v_GRID_c_r - v_POI_c_r
        struct[0].g[29,0] = -1.96237550602401*i_POI_a_i - 1.90429395893589*i_POI_a_r - 1.96237550602406*i_POI_b_i - 1.90429395893597*i_POI_b_r + 8.99352976113174*i_POI_c_i + 16.3488065237228*i_POI_c_r - 1.91798392779206*i_POImv_b_i - 3.19497213887036*i_POImv_b_r + 1.91798392779192*i_POImv_c_i + 3.19497213887045*i_POImv_c_r - 0.0382049854269434*i_W1lv_a_i - 0.0636395371726713*i_W1lv_a_r - 0.0382049854269442*i_W1lv_b_i - 0.0636395371726737*i_W1lv_b_r + 0.0764099708538875*i_W1lv_c_i + 0.127279074345345*i_W1lv_c_r - 1.91805727135935*i_W1mv_b_i - 3.19498294936915*i_W1mv_b_r + 1.91805727135921*i_W1mv_c_i + 3.19498294936923*i_W1mv_c_r - 0.0382048231043602*i_W2lv_a_i - 0.0636395132474596*i_W2lv_a_r - 0.038204823104361*i_W2lv_b_i - 0.0636395132474621*i_W2lv_b_r + 0.0764096462087212*i_W2lv_c_i + 0.127279026494922*i_W2lv_c_r - 1.91804912205613*i_W2mv_b_i - 3.19498174821894*i_W2mv_b_r + 1.91804912205598*i_W2mv_c_i + 3.19498174821902*i_W2mv_c_r - 0.0382043361371476*i_W3lv_a_i - 0.063639441471337*i_W3lv_a_r - 0.0382043361371484*i_W3lv_b_i - 0.0636394414713394*i_W3lv_b_r + 0.0764086722742961*i_W3lv_c_i + 0.127278882942676*i_W3lv_c_r - 1.91802467417341*i_W3mv_b_i - 3.19497814474383*i_W3mv_b_r + 1.91802467417326*i_W3mv_c_i + 3.19497814474392*i_W3mv_c_r - 0.124162742246184*v_GRID_a_i + 0.0330297796399061*v_GRID_a_r - 0.124162742246189*v_GRID_b_i + 0.0330297796399062*v_GRID_b_r + 0.876104930717239*v_GRID_c_i + 0.032866807135454*v_GRID_c_r - v_POI_c_i
        struct[0].g[30,0] = -3.19498294936899*i_POI_a_i + 1.91805727135919*i_POI_a_r + 3.19498294936919*i_POI_b_i - 1.91805727135919*i_POI_b_r - 3.08738963687029*i_POImv_a_i + 11.6035242966407*i_POImv_a_r + 1.79198075607073*i_POImv_b_i + 10.1957014483333*i_POImv_b_r + 1.79198075607047*i_POImv_c_i + 10.1957014483333*i_POImv_c_r - 0.100328108879386*i_W1lv_a_i + 0.0318681229122168*i_W1lv_a_r + 0.100328108879391*i_W1lv_b_i - 0.0318681229122152*i_W1lv_b_r - 3.34841422289852*i_W1mv_a_i + 11.9248072396495*i_W1mv_a_r + 1.68849540047563*i_W1mv_b_i + 10.3248881664377*i_W1mv_b_r + 1.68849540047537*i_W1mv_c_i + 10.3248881664376*i_W1mv_c_r - 0.09928229701422*i_W2lv_a_i + 0.0305931173773991*i_W2lv_a_r + 0.0992822970142252*i_W2lv_b_i - 0.0305931173773975*i_W2lv_b_r - 3.26107847080994*i_W2mv_a_i + 11.81799648295*i_W2mv_a_r + 1.72332682544462*i_W2mv_b_i + 10.2820882609335*i_W2mv_b_r + 1.72332682544436*i_W2mv_c_i + 10.2820882609334*i_W2mv_c_r - 0.0982363113431474*i_W3lv_a_i + 0.0293176867112793*i_W3lv_a_r + 0.0982363113431526*i_W3lv_b_i - 0.0293176867112777*i_W3lv_b_r - 3.17407051442937*i_W3mv_a_i + 11.7109010138509*i_W3mv_a_r + 1.75782172888933*i_W3mv_b_i + 10.2390249864793*i_W3mv_b_r + 1.75782172888906*i_W3mv_c_i + 10.2390249864793*i_W3mv_c_r + 4.30097396370904e-5*v_GRID_a_i + 0.175093364713316*v_GRID_a_r - 4.30097396321211e-5*v_GRID_b_i - 0.175093364713324*v_GRID_b_r - 3.95107998085596e-17*v_GRID_c_i + 3.80502101367139e-17*v_GRID_c_r - v_W1mv_a_r
        struct[0].g[31,0] = 1.91805727135919*i_POI_a_i + 3.19498294936899*i_POI_a_r - 1.91805727135919*i_POI_b_i - 3.19498294936919*i_POI_b_r + 11.6035242966407*i_POImv_a_i + 3.08738963687029*i_POImv_a_r + 10.1957014483333*i_POImv_b_i - 1.79198075607073*i_POImv_b_r + 10.1957014483333*i_POImv_c_i - 1.79198075607047*i_POImv_c_r + 0.0318681229122168*i_W1lv_a_i + 0.100328108879386*i_W1lv_a_r - 0.0318681229122152*i_W1lv_b_i - 0.100328108879391*i_W1lv_b_r + 11.9248072396495*i_W1mv_a_i + 3.34841422289852*i_W1mv_a_r + 10.3248881664377*i_W1mv_b_i - 1.68849540047563*i_W1mv_b_r + 10.3248881664376*i_W1mv_c_i - 1.68849540047537*i_W1mv_c_r + 0.0305931173773991*i_W2lv_a_i + 0.09928229701422*i_W2lv_a_r - 0.0305931173773975*i_W2lv_b_i - 0.0992822970142252*i_W2lv_b_r + 11.81799648295*i_W2mv_a_i + 3.26107847080994*i_W2mv_a_r + 10.2820882609335*i_W2mv_b_i - 1.72332682544462*i_W2mv_b_r + 10.2820882609334*i_W2mv_c_i - 1.72332682544436*i_W2mv_c_r + 0.0293176867112793*i_W3lv_a_i + 0.0982363113431474*i_W3lv_a_r - 0.0293176867112777*i_W3lv_b_i - 0.0982363113431526*i_W3lv_b_r + 11.7109010138509*i_W3mv_a_i + 3.17407051442937*i_W3mv_a_r + 10.2390249864793*i_W3mv_b_i - 1.75782172888933*i_W3mv_b_r + 10.2390249864793*i_W3mv_c_i - 1.75782172888906*i_W3mv_c_r + 0.175093364713316*v_GRID_a_i - 4.30097396370904e-5*v_GRID_a_r - 0.175093364713324*v_GRID_b_i + 4.30097396321211e-5*v_GRID_b_r + 3.80502101367139e-17*v_GRID_c_i + 3.95107998085596e-17*v_GRID_c_r - v_W1mv_a_i
        struct[0].g[32,0] = -3.19498294936915*i_POI_b_i + 1.91805727135931*i_POI_b_r + 3.19498294936901*i_POI_c_i - 1.91805727135932*i_POI_c_r + 1.7919807560709*i_POImv_a_i + 10.1957014483335*i_POImv_a_r - 3.0873896368701*i_POImv_b_i + 11.603524296641*i_POImv_b_r + 1.79198075607072*i_POImv_c_i + 10.1957014483334*i_POImv_c_r - 0.100328108879391*i_W1lv_b_i + 0.0318681229122188*i_W1lv_b_r + 0.100328108879387*i_W1lv_c_i - 0.0318681229122203*i_W1lv_c_r + 1.68849540047579*i_W1mv_a_i + 10.3248881664379*i_W1mv_a_r - 3.34841422289833*i_W1mv_b_i + 11.9248072396498*i_W1mv_b_r + 1.68849540047561*i_W1mv_c_i + 10.3248881664378*i_W1mv_c_r - 0.099282297014225*i_W2lv_b_i + 0.0305931173774011*i_W2lv_b_r + 0.0992822970142214*i_W2lv_c_i - 0.0305931173774025*i_W2lv_c_r + 1.72332682544478*i_W2mv_a_i + 10.2820882609337*i_W2mv_a_r - 3.26107847080975*i_W2mv_b_i + 11.8179964829504*i_W2mv_b_r + 1.72332682544462*i_W2mv_c_i + 10.2820882609336*i_W2mv_c_r - 0.0982363113431523*i_W3lv_b_i + 0.0293176867112812*i_W3lv_b_r + 0.0982363113431489*i_W3lv_c_i - 0.0293176867112828*i_W3lv_c_r + 1.7578217288895*i_W3mv_a_i + 10.2390249864795*i_W3mv_a_r - 3.17407051442918*i_W3mv_b_i + 11.7109010138513*i_W3mv_b_r + 1.75782172888933*i_W3mv_c_i + 10.2390249864795*i_W3mv_c_r + 4.89578301787291e-18*v_GRID_a_i + 2.93583452294173e-18*v_GRID_a_r + 4.30097396379464e-5*v_GRID_b_i + 0.175093364713325*v_GRID_b_r - 4.30097396419938e-5*v_GRID_c_i - 0.17509336471332*v_GRID_c_r - v_W1mv_b_r
        struct[0].g[33,0] = 1.91805727135931*i_POI_b_i + 3.19498294936915*i_POI_b_r - 1.91805727135932*i_POI_c_i - 3.19498294936901*i_POI_c_r + 10.1957014483335*i_POImv_a_i - 1.7919807560709*i_POImv_a_r + 11.603524296641*i_POImv_b_i + 3.0873896368701*i_POImv_b_r + 10.1957014483334*i_POImv_c_i - 1.79198075607072*i_POImv_c_r + 0.0318681229122188*i_W1lv_b_i + 0.100328108879391*i_W1lv_b_r - 0.0318681229122203*i_W1lv_c_i - 0.100328108879387*i_W1lv_c_r + 10.3248881664379*i_W1mv_a_i - 1.68849540047579*i_W1mv_a_r + 11.9248072396498*i_W1mv_b_i + 3.34841422289833*i_W1mv_b_r + 10.3248881664378*i_W1mv_c_i - 1.68849540047561*i_W1mv_c_r + 0.0305931173774011*i_W2lv_b_i + 0.099282297014225*i_W2lv_b_r - 0.0305931173774025*i_W2lv_c_i - 0.0992822970142214*i_W2lv_c_r + 10.2820882609337*i_W2mv_a_i - 1.72332682544478*i_W2mv_a_r + 11.8179964829504*i_W2mv_b_i + 3.26107847080975*i_W2mv_b_r + 10.2820882609336*i_W2mv_c_i - 1.72332682544462*i_W2mv_c_r + 0.0293176867112812*i_W3lv_b_i + 0.0982363113431523*i_W3lv_b_r - 0.0293176867112828*i_W3lv_c_i - 0.0982363113431489*i_W3lv_c_r + 10.2390249864795*i_W3mv_a_i - 1.7578217288895*i_W3mv_a_r + 11.7109010138513*i_W3mv_b_i + 3.17407051442918*i_W3mv_b_r + 10.2390249864795*i_W3mv_c_i - 1.75782172888933*i_W3mv_c_r + 2.93583452294173e-18*v_GRID_a_i - 4.89578301787291e-18*v_GRID_a_r + 0.175093364713325*v_GRID_b_i - 4.30097396379464e-5*v_GRID_b_r - 0.17509336471332*v_GRID_c_i + 4.30097396419938e-5*v_GRID_c_r - v_W1mv_b_i
        struct[0].g[34,0] = 3.19498294936927*i_POI_a_i - 1.91805727135924*i_POI_a_r - 3.19498294936936*i_POI_c_i + 1.91805727135924*i_POI_c_r + 1.79198075607079*i_POImv_a_i + 10.1957014483334*i_POImv_a_r + 1.79198075607092*i_POImv_b_i + 10.1957014483334*i_POImv_b_r - 3.08738963687035*i_POImv_c_i + 11.6035242966408*i_POImv_c_r + 0.100328108879394*i_W1lv_a_i - 0.0318681229122163*i_W1lv_a_r - 0.100328108879396*i_W1lv_c_i + 0.0318681229122155*i_W1lv_c_r + 1.68849540047567*i_W1mv_a_i + 10.3248881664378*i_W1mv_a_r + 1.68849540047581*i_W1mv_b_i + 10.3248881664378*i_W1mv_b_r - 3.34841422289858*i_W1mv_c_i + 11.9248072396496*i_W1mv_c_r + 0.0992822970142275*i_W2lv_a_i - 0.0305931173773985*i_W2lv_a_r - 0.0992822970142302*i_W2lv_c_i + 0.0305931173773979*i_W2lv_c_r + 1.72332682544468*i_W2mv_a_i + 10.2820882609336*i_W2mv_a_r + 1.72332682544481*i_W2mv_b_i + 10.2820882609336*i_W2mv_b_r - 3.26107847081*i_W2mv_c_i + 11.8179964829502*i_W2mv_c_r + 0.098236311343155*i_W3lv_a_i - 0.0293176867112786*i_W3lv_a_r - 0.0982363113431575*i_W3lv_c_i + 0.0293176867112779*i_W3lv_c_r + 1.75782172888938*i_W3mv_a_i + 10.2390249864794*i_W3mv_a_r + 1.75782172888952*i_W3mv_b_i + 10.2390249864795*i_W3mv_b_r - 3.17407051442944*i_W3mv_c_i + 11.7109010138511*i_W3mv_c_r - 4.30097396325989e-5*v_GRID_a_i - 0.175093364713328*v_GRID_a_r + 3.10996770759589e-17*v_GRID_b_i - 4.08938066674102e-17*v_GRID_b_r + 4.30097396300291e-5*v_GRID_c_i + 0.175093364713332*v_GRID_c_r - v_W1mv_c_r
        struct[0].g[35,0] = -1.91805727135924*i_POI_a_i - 3.19498294936927*i_POI_a_r + 1.91805727135924*i_POI_c_i + 3.19498294936936*i_POI_c_r + 10.1957014483334*i_POImv_a_i - 1.79198075607079*i_POImv_a_r + 10.1957014483334*i_POImv_b_i - 1.79198075607092*i_POImv_b_r + 11.6035242966408*i_POImv_c_i + 3.08738963687035*i_POImv_c_r - 0.0318681229122163*i_W1lv_a_i - 0.100328108879394*i_W1lv_a_r + 0.0318681229122155*i_W1lv_c_i + 0.100328108879396*i_W1lv_c_r + 10.3248881664378*i_W1mv_a_i - 1.68849540047567*i_W1mv_a_r + 10.3248881664378*i_W1mv_b_i - 1.68849540047581*i_W1mv_b_r + 11.9248072396496*i_W1mv_c_i + 3.34841422289858*i_W1mv_c_r - 0.0305931173773985*i_W2lv_a_i - 0.0992822970142275*i_W2lv_a_r + 0.0305931173773979*i_W2lv_c_i + 0.0992822970142302*i_W2lv_c_r + 10.2820882609336*i_W2mv_a_i - 1.72332682544468*i_W2mv_a_r + 10.2820882609336*i_W2mv_b_i - 1.72332682544481*i_W2mv_b_r + 11.8179964829502*i_W2mv_c_i + 3.26107847081*i_W2mv_c_r - 0.0293176867112786*i_W3lv_a_i - 0.098236311343155*i_W3lv_a_r + 0.0293176867112779*i_W3lv_c_i + 0.0982363113431575*i_W3lv_c_r + 10.2390249864794*i_W3mv_a_i - 1.75782172888938*i_W3mv_a_r + 10.2390249864795*i_W3mv_b_i - 1.75782172888952*i_W3mv_b_r + 11.7109010138511*i_W3mv_c_i + 3.17407051442944*i_W3mv_c_r - 0.175093364713328*v_GRID_a_i + 4.30097396325989e-5*v_GRID_a_r - 4.08938066674102e-17*v_GRID_b_i - 3.10996770759589e-17*v_GRID_b_r + 0.175093364713332*v_GRID_c_i - 4.30097396300291e-5*v_GRID_c_r - v_W1mv_c_i
        struct[0].g[36,0] = -3.19498174821879*i_POI_a_i + 1.91804912205597*i_POI_a_r + 3.19498174821898*i_POI_b_i - 1.91804912205596*i_POI_b_r - 3.08755280951041*i_POImv_a_i + 11.60338540301*i_POImv_a_r + 1.79181314887337*i_POImv_b_i + 10.1955728673526*i_POImv_b_r + 1.79181314887312*i_POImv_c_i + 10.1955728673526*i_POImv_c_r - 0.0992822970142201*i_W1lv_a_i + 0.0305931173773991*i_W1lv_a_r + 0.0992822970142253*i_W1lv_b_i - 0.0305931173773975*i_W1lv_b_r - 3.26107847080993*i_W1mv_a_i + 11.81799648295*i_W1mv_a_r + 1.72332682544462*i_W1mv_b_i + 10.2820882609335*i_W1mv_b_r + 1.72332682544436*i_W1mv_c_i + 10.2820882609334*i_W1mv_c_r - 0.0992822101112667*i_W2lv_a_i + 0.030592904811748*i_W2lv_a_r + 0.0992822101112719*i_W2lv_b_i - 0.0305929048117464*i_W2lv_b_r - 3.26124236866395*i_W2mv_a_i + 11.8178541267502*i_W2mv_a_r + 1.72315856468247*i_W2mv_b_i + 10.2819565764585*i_W2mv_b_r + 1.72315856468222*i_W2mv_c_i + 10.2819565764584*i_W2mv_c_r - 0.0982362237268541*i_W3lv_a_i + 0.0293174777213171*i_W3lv_a_r + 0.0982362237268593*i_W3lv_b_i - 0.0293174777213155*i_W3lv_b_r - 3.17423405384011*i_W3mv_a_i + 11.7107603897954*i_W3mv_a_r + 1.75765379075767*i_W3mv_b_i + 10.2388948546334*i_W3mv_b_r + 1.75765379075741*i_W3mv_c_i + 10.2388948546333*i_W3mv_c_r + 4.27104401568212e-5*v_GRID_a_i + 0.175093119317178*v_GRID_a_r - 4.27104401518103e-5*v_GRID_b_i - 0.175093119317186*v_GRID_b_r - 3.95108094457718e-17*v_GRID_c_i + 3.8050089267765e-17*v_GRID_c_r - v_W2mv_a_r
        struct[0].g[37,0] = 1.91804912205597*i_POI_a_i + 3.19498174821879*i_POI_a_r - 1.91804912205596*i_POI_b_i - 3.19498174821898*i_POI_b_r + 11.60338540301*i_POImv_a_i + 3.08755280951041*i_POImv_a_r + 10.1955728673526*i_POImv_b_i - 1.79181314887337*i_POImv_b_r + 10.1955728673526*i_POImv_c_i - 1.79181314887312*i_POImv_c_r + 0.0305931173773991*i_W1lv_a_i + 0.0992822970142201*i_W1lv_a_r - 0.0305931173773975*i_W1lv_b_i - 0.0992822970142253*i_W1lv_b_r + 11.81799648295*i_W1mv_a_i + 3.26107847080993*i_W1mv_a_r + 10.2820882609335*i_W1mv_b_i - 1.72332682544462*i_W1mv_b_r + 10.2820882609334*i_W1mv_c_i - 1.72332682544436*i_W1mv_c_r + 0.030592904811748*i_W2lv_a_i + 0.0992822101112667*i_W2lv_a_r - 0.0305929048117464*i_W2lv_b_i - 0.0992822101112719*i_W2lv_b_r + 11.8178541267502*i_W2mv_a_i + 3.26124236866395*i_W2mv_a_r + 10.2819565764585*i_W2mv_b_i - 1.72315856468247*i_W2mv_b_r + 10.2819565764584*i_W2mv_c_i - 1.72315856468222*i_W2mv_c_r + 0.0293174777213171*i_W3lv_a_i + 0.0982362237268541*i_W3lv_a_r - 0.0293174777213155*i_W3lv_b_i - 0.0982362237268593*i_W3lv_b_r + 11.7107603897954*i_W3mv_a_i + 3.17423405384011*i_W3mv_a_r + 10.2388948546334*i_W3mv_b_i - 1.75765379075767*i_W3mv_b_r + 10.2388948546333*i_W3mv_c_i - 1.75765379075741*i_W3mv_c_r + 0.175093119317178*v_GRID_a_i - 4.27104401568212e-5*v_GRID_a_r - 0.175093119317186*v_GRID_b_i + 4.27104401518103e-5*v_GRID_b_r + 3.8050089267765e-17*v_GRID_c_i + 3.95108094457718e-17*v_GRID_c_r - v_W2mv_a_i
        struct[0].g[38,0] = -3.19498174821894*i_POI_b_i + 1.91804912205608*i_POI_b_r + 3.1949817482188*i_POI_c_i - 1.9180491220561*i_POI_c_r + 1.79181314887354*i_POImv_a_i + 10.1955728673528*i_POImv_a_r - 3.08755280951023*i_POImv_b_i + 11.6033854030104*i_POImv_b_r + 1.79181314887337*i_POImv_c_i + 10.1955728673527*i_POImv_c_r - 0.099282297014225*i_W1lv_b_i + 0.0305931173774011*i_W1lv_b_r + 0.0992822970142215*i_W1lv_c_i - 0.0305931173774025*i_W1lv_c_r + 1.72332682544478*i_W1mv_a_i + 10.2820882609337*i_W1mv_a_r - 3.26107847080975*i_W1mv_b_i + 11.8179964829504*i_W1mv_b_r + 1.72332682544461*i_W1mv_c_i + 10.2820882609336*i_W1mv_c_r - 0.0992822101112716*i_W2lv_b_i + 0.0305929048117499*i_W2lv_b_r + 0.0992822101112681*i_W2lv_c_i - 0.0305929048117514*i_W2lv_c_r + 1.72315856468264*i_W2mv_a_i + 10.2819565764587*i_W2mv_a_r - 3.26124236866376*i_W2mv_b_i + 11.8178541267506*i_W2mv_b_r + 1.72315856468247*i_W2mv_c_i + 10.2819565764586*i_W2mv_c_r - 0.098236223726859*i_W3lv_b_i + 0.029317477721319*i_W3lv_b_r + 0.0982362237268555*i_W3lv_c_i - 0.0293174777213205*i_W3lv_c_r + 1.75765379075784*i_W3mv_a_i + 10.2388948546336*i_W3mv_a_r - 3.17423405383993*i_W3mv_b_i + 11.7107603897957*i_W3mv_b_r + 1.75765379075766*i_W3mv_c_i + 10.2388948546335*i_W3mv_c_r + 4.8957711368811e-18*v_GRID_a_i + 2.93583877411288e-18*v_GRID_a_r + 4.27104401575523e-5*v_GRID_b_i + 0.175093119317187*v_GRID_b_r - 4.27104401616275e-5*v_GRID_c_i - 0.175093119317182*v_GRID_c_r - v_W2mv_b_r
        struct[0].g[39,0] = 1.91804912205608*i_POI_b_i + 3.19498174821894*i_POI_b_r - 1.9180491220561*i_POI_c_i - 3.1949817482188*i_POI_c_r + 10.1955728673528*i_POImv_a_i - 1.79181314887354*i_POImv_a_r + 11.6033854030104*i_POImv_b_i + 3.08755280951023*i_POImv_b_r + 10.1955728673527*i_POImv_c_i - 1.79181314887337*i_POImv_c_r + 0.0305931173774011*i_W1lv_b_i + 0.099282297014225*i_W1lv_b_r - 0.0305931173774025*i_W1lv_c_i - 0.0992822970142215*i_W1lv_c_r + 10.2820882609337*i_W1mv_a_i - 1.72332682544478*i_W1mv_a_r + 11.8179964829504*i_W1mv_b_i + 3.26107847080975*i_W1mv_b_r + 10.2820882609336*i_W1mv_c_i - 1.72332682544461*i_W1mv_c_r + 0.0305929048117499*i_W2lv_b_i + 0.0992822101112716*i_W2lv_b_r - 0.0305929048117514*i_W2lv_c_i - 0.0992822101112681*i_W2lv_c_r + 10.2819565764587*i_W2mv_a_i - 1.72315856468264*i_W2mv_a_r + 11.8178541267506*i_W2mv_b_i + 3.26124236866376*i_W2mv_b_r + 10.2819565764586*i_W2mv_c_i - 1.72315856468247*i_W2mv_c_r + 0.029317477721319*i_W3lv_b_i + 0.098236223726859*i_W3lv_b_r - 0.0293174777213205*i_W3lv_c_i - 0.0982362237268555*i_W3lv_c_r + 10.2388948546336*i_W3mv_a_i - 1.75765379075784*i_W3mv_a_r + 11.7107603897957*i_W3mv_b_i + 3.17423405383993*i_W3mv_b_r + 10.2388948546335*i_W3mv_c_i - 1.75765379075766*i_W3mv_c_r + 2.93583877411288e-18*v_GRID_a_i - 4.8957711368811e-18*v_GRID_a_r + 0.175093119317187*v_GRID_b_i - 4.27104401575523e-5*v_GRID_b_r - 0.175093119317182*v_GRID_c_i + 4.27104401616275e-5*v_GRID_c_r - v_W2mv_b_i
        struct[0].g[40,0] = 3.19498174821906*i_POI_a_i - 1.91804912205602*i_POI_a_r - 3.19498174821916*i_POI_c_i + 1.91804912205601*i_POI_c_r + 1.79181314887342*i_POImv_a_i + 10.1955728673527*i_POImv_a_r + 1.79181314887355*i_POImv_b_i + 10.1955728673527*i_POImv_b_r - 3.08755280951048*i_POImv_c_i + 11.6033854030101*i_POImv_c_r + 0.0992822970142276*i_W1lv_a_i - 0.0305931173773985*i_W1lv_a_r - 0.0992822970142302*i_W1lv_c_i + 0.0305931173773978*i_W1lv_c_r + 1.72332682544467*i_W1mv_a_i + 10.2820882609336*i_W1mv_a_r + 1.7233268254448*i_W1mv_b_i + 10.2820882609336*i_W1mv_b_r - 3.26107847081*i_W1mv_c_i + 11.8179964829501*i_W1mv_c_r + 0.0992822101112742*i_W2lv_a_i - 0.0305929048117474*i_W2lv_a_r - 0.0992822101112768*i_W2lv_c_i + 0.0305929048117467*i_W2lv_c_r + 1.72315856468253*i_W2mv_a_i + 10.2819565764586*i_W2mv_a_r + 1.72315856468266*i_W2mv_b_i + 10.2819565764586*i_W2mv_b_r - 3.26124236866401*i_W2mv_c_i + 11.8178541267503*i_W2mv_c_r + 0.0982362237268616*i_W3lv_a_i - 0.0293174777213164*i_W3lv_a_r - 0.0982362237268642*i_W3lv_c_i + 0.0293174777213157*i_W3lv_c_r + 1.75765379075772*i_W3mv_a_i + 10.2388948546335*i_W3mv_a_r + 1.75765379075786*i_W3mv_b_i + 10.2388948546335*i_W3mv_b_r - 3.17423405384018*i_W3mv_c_i + 11.7107603897955*i_W3mv_c_r - 4.2710440152288e-5*v_GRID_a_i - 0.175093119317191*v_GRID_a_r + 3.1099703364806e-17*v_GRID_b_i - 4.08936961867526e-17*v_GRID_b_r + 4.27104401497182e-5*v_GRID_c_i + 0.175093119317194*v_GRID_c_r - v_W2mv_c_r
        struct[0].g[41,0] = -1.91804912205602*i_POI_a_i - 3.19498174821906*i_POI_a_r + 1.91804912205601*i_POI_c_i + 3.19498174821916*i_POI_c_r + 10.1955728673527*i_POImv_a_i - 1.79181314887342*i_POImv_a_r + 10.1955728673527*i_POImv_b_i - 1.79181314887355*i_POImv_b_r + 11.6033854030101*i_POImv_c_i + 3.08755280951048*i_POImv_c_r - 0.0305931173773985*i_W1lv_a_i - 0.0992822970142276*i_W1lv_a_r + 0.0305931173773978*i_W1lv_c_i + 0.0992822970142302*i_W1lv_c_r + 10.2820882609336*i_W1mv_a_i - 1.72332682544467*i_W1mv_a_r + 10.2820882609336*i_W1mv_b_i - 1.7233268254448*i_W1mv_b_r + 11.8179964829501*i_W1mv_c_i + 3.26107847081*i_W1mv_c_r - 0.0305929048117474*i_W2lv_a_i - 0.0992822101112742*i_W2lv_a_r + 0.0305929048117467*i_W2lv_c_i + 0.0992822101112768*i_W2lv_c_r + 10.2819565764586*i_W2mv_a_i - 1.72315856468253*i_W2mv_a_r + 10.2819565764586*i_W2mv_b_i - 1.72315856468266*i_W2mv_b_r + 11.8178541267503*i_W2mv_c_i + 3.26124236866401*i_W2mv_c_r - 0.0293174777213164*i_W3lv_a_i - 0.0982362237268616*i_W3lv_a_r + 0.0293174777213157*i_W3lv_c_i + 0.0982362237268642*i_W3lv_c_r + 10.2388948546335*i_W3mv_a_i - 1.75765379075772*i_W3mv_a_r + 10.2388948546335*i_W3mv_b_i - 1.75765379075786*i_W3mv_b_r + 11.7107603897955*i_W3mv_c_i + 3.17423405384018*i_W3mv_c_r - 0.175093119317191*v_GRID_a_i + 4.2710440152288e-5*v_GRID_a_r - 4.08936961867526e-17*v_GRID_b_i - 3.1099703364806e-17*v_GRID_b_r + 0.175093119317194*v_GRID_c_i - 4.27104401497182e-5*v_GRID_c_r - v_W2mv_c_i
        struct[0].g[42,0] = -3.19497814474368*i_POI_a_i + 1.91802467417325*i_POI_a_r + 3.19497814474388*i_POI_b_i - 1.91802467417324*i_POI_b_r - 3.08804231916211*i_POImv_a_i + 11.6029687203683*i_POImv_a_r + 1.79131033552716*i_POImv_b_i + 10.1951871226168*i_POImv_b_r + 1.7913103355269*i_POImv_c_i + 10.1951871226167*i_POImv_c_r - 0.0982363113431474*i_W1lv_a_i + 0.0293176867112792*i_W1lv_a_r + 0.0982363113431526*i_W1lv_b_i - 0.0293176867112776*i_W1lv_b_r - 3.17407051442937*i_W1mv_a_i + 11.7109010138509*i_W1mv_a_r + 1.75782172888933*i_W1mv_b_i + 10.2390249864793*i_W1mv_b_r + 1.75782172888907*i_W1mv_c_i + 10.2390249864793*i_W1mv_c_r - 0.0982362237268541*i_W2lv_a_i + 0.029317477721317*i_W2lv_a_r + 0.0982362237268593*i_W2lv_b_i - 0.0293174777213155*i_W2lv_b_r - 3.17423405384011*i_W2mv_a_i + 11.7107603897953*i_W2mv_a_r + 1.75765379075767*i_W2mv_b_i + 10.2388948546334*i_W2mv_b_r + 1.75765379075741*i_W2mv_c_i + 10.2388948546333*i_W2mv_c_r - 0.0982359608775054*i_W3lv_a_i + 0.0293168507523162*i_W3lv_a_r + 0.0982359608775105*i_W3lv_b_i - 0.0293168507523146*i_W3lv_b_r - 3.17472466374498*i_W3mv_a_i + 11.7103385159093*i_W3mv_a_r + 1.75714998466652*i_W3mv_b_i + 10.2385044573318*i_W3mv_b_r + 1.75714998466626*i_W3mv_c_i + 10.2385044573317*i_W3mv_c_r + 4.18125433939492e-5*v_GRID_a_i + 0.17509238312843*v_GRID_a_r - 4.18125433889522e-5*v_GRID_b_i - 0.175092383128438*v_GRID_b_r - 3.9510838356968e-17*v_GRID_c_i + 3.80497266612242e-17*v_GRID_c_r - v_W3mv_a_r
        struct[0].g[43,0] = 1.91802467417325*i_POI_a_i + 3.19497814474368*i_POI_a_r - 1.91802467417324*i_POI_b_i - 3.19497814474388*i_POI_b_r + 11.6029687203683*i_POImv_a_i + 3.08804231916211*i_POImv_a_r + 10.1951871226168*i_POImv_b_i - 1.79131033552716*i_POImv_b_r + 10.1951871226167*i_POImv_c_i - 1.7913103355269*i_POImv_c_r + 0.0293176867112792*i_W1lv_a_i + 0.0982363113431474*i_W1lv_a_r - 0.0293176867112776*i_W1lv_b_i - 0.0982363113431526*i_W1lv_b_r + 11.7109010138509*i_W1mv_a_i + 3.17407051442937*i_W1mv_a_r + 10.2390249864793*i_W1mv_b_i - 1.75782172888933*i_W1mv_b_r + 10.2390249864793*i_W1mv_c_i - 1.75782172888907*i_W1mv_c_r + 0.029317477721317*i_W2lv_a_i + 0.0982362237268541*i_W2lv_a_r - 0.0293174777213155*i_W2lv_b_i - 0.0982362237268593*i_W2lv_b_r + 11.7107603897953*i_W2mv_a_i + 3.17423405384011*i_W2mv_a_r + 10.2388948546334*i_W2mv_b_i - 1.75765379075767*i_W2mv_b_r + 10.2388948546333*i_W2mv_c_i - 1.75765379075741*i_W2mv_c_r + 0.0293168507523162*i_W3lv_a_i + 0.0982359608775054*i_W3lv_a_r - 0.0293168507523146*i_W3lv_b_i - 0.0982359608775105*i_W3lv_b_r + 11.7103385159093*i_W3mv_a_i + 3.17472466374498*i_W3mv_a_r + 10.2385044573318*i_W3mv_b_i - 1.75714998466652*i_W3mv_b_r + 10.2385044573317*i_W3mv_c_i - 1.75714998466626*i_W3mv_c_r + 0.17509238312843*v_GRID_a_i - 4.18125433939492e-5*v_GRID_a_r - 0.175092383128438*v_GRID_b_i + 4.18125433889522e-5*v_GRID_b_r + 3.80497266612242e-17*v_GRID_c_i + 3.9510838356968e-17*v_GRID_c_r - v_W3mv_a_i
        struct[0].g[44,0] = -3.19497814474384*i_POI_b_i + 1.91802467417336*i_POI_b_r + 3.1949781447437*i_POI_c_i - 1.91802467417338*i_POI_c_r + 1.79131033552732*i_POImv_a_i + 10.1951871226169*i_POImv_a_r - 3.08804231916193*i_POImv_b_i + 11.6029687203686*i_POImv_b_r + 1.79131033552715*i_POImv_c_i + 10.1951871226169*i_POImv_c_r - 0.0982363113431524*i_W1lv_b_i + 0.0293176867112812*i_W1lv_b_r + 0.0982363113431489*i_W1lv_c_i - 0.0293176867112826*i_W1lv_c_r + 1.7578217288895*i_W1mv_a_i + 10.2390249864795*i_W1mv_a_r - 3.17407051442919*i_W1mv_b_i + 11.7109010138513*i_W1mv_b_r + 1.75782172888932*i_W1mv_c_i + 10.2390249864794*i_W1mv_c_r - 0.098236223726859*i_W2lv_b_i + 0.029317477721319*i_W2lv_b_r + 0.0982362237268555*i_W2lv_c_i - 0.0293174777213204*i_W2lv_c_r + 1.75765379075784*i_W2mv_a_i + 10.2388948546336*i_W2mv_a_r - 3.17423405383993*i_W2mv_b_i + 11.7107603897957*i_W2mv_b_r + 1.75765379075766*i_W2mv_c_i + 10.2388948546335*i_W2mv_c_r - 0.0982359608775103*i_W3lv_b_i + 0.0293168507523179*i_W3lv_b_r + 0.0982359608775067*i_W3lv_c_i - 0.0293168507523194*i_W3lv_c_r + 1.75714998466669*i_W3mv_a_i + 10.238504457332*i_W3mv_a_r - 3.1747246637448*i_W3mv_b_i + 11.7103385159096*i_W3mv_b_r + 1.75714998466651*i_W3mv_c_i + 10.2385044573319*i_W3mv_c_r + 4.89573549392443e-18*v_GRID_a_i + 2.93585152757382e-18*v_GRID_a_r + 4.18125433946109e-5*v_GRID_b_i + 0.175092383128439*v_GRID_b_r - 4.18125433986861e-5*v_GRID_c_i - 0.175092383128434*v_GRID_c_r - v_W3mv_b_r
        struct[0].g[45,0] = 1.91802467417336*i_POI_b_i + 3.19497814474384*i_POI_b_r - 1.91802467417338*i_POI_c_i - 3.1949781447437*i_POI_c_r + 10.1951871226169*i_POImv_a_i - 1.79131033552732*i_POImv_a_r + 11.6029687203686*i_POImv_b_i + 3.08804231916193*i_POImv_b_r + 10.1951871226169*i_POImv_c_i - 1.79131033552715*i_POImv_c_r + 0.0293176867112812*i_W1lv_b_i + 0.0982363113431524*i_W1lv_b_r - 0.0293176867112826*i_W1lv_c_i - 0.0982363113431489*i_W1lv_c_r + 10.2390249864795*i_W1mv_a_i - 1.7578217288895*i_W1mv_a_r + 11.7109010138513*i_W1mv_b_i + 3.17407051442919*i_W1mv_b_r + 10.2390249864794*i_W1mv_c_i - 1.75782172888932*i_W1mv_c_r + 0.029317477721319*i_W2lv_b_i + 0.098236223726859*i_W2lv_b_r - 0.0293174777213204*i_W2lv_c_i - 0.0982362237268555*i_W2lv_c_r + 10.2388948546336*i_W2mv_a_i - 1.75765379075784*i_W2mv_a_r + 11.7107603897957*i_W2mv_b_i + 3.17423405383993*i_W2mv_b_r + 10.2388948546335*i_W2mv_c_i - 1.75765379075766*i_W2mv_c_r + 0.0293168507523179*i_W3lv_b_i + 0.0982359608775103*i_W3lv_b_r - 0.0293168507523194*i_W3lv_c_i - 0.0982359608775067*i_W3lv_c_r + 10.238504457332*i_W3mv_a_i - 1.75714998466669*i_W3mv_a_r + 11.7103385159096*i_W3mv_b_i + 3.1747246637448*i_W3mv_b_r + 10.2385044573319*i_W3mv_c_i - 1.75714998466651*i_W3mv_c_r + 2.93585152757382e-18*v_GRID_a_i - 4.89573549392443e-18*v_GRID_a_r + 0.175092383128439*v_GRID_b_i - 4.18125433946109e-5*v_GRID_b_r - 0.175092383128434*v_GRID_c_i + 4.18125433986861e-5*v_GRID_c_r - v_W3mv_b_i
        struct[0].g[46,0] = 3.19497814474395*i_POI_a_i - 1.9180246741733*i_POI_a_r - 3.19497814474405*i_POI_c_i + 1.91802467417329*i_POI_c_r + 1.79131033552721*i_POImv_a_i + 10.1951871226168*i_POImv_a_r + 1.79131033552733*i_POImv_b_i + 10.1951871226169*i_POImv_b_r - 3.08804231916218*i_POImv_c_i + 11.6029687203684*i_POImv_c_r + 0.098236311343155*i_W1lv_a_i - 0.0293176867112787*i_W1lv_a_r - 0.0982363113431576*i_W1lv_c_i + 0.0293176867112779*i_W1lv_c_r + 1.75782172888938*i_W1mv_a_i + 10.2390249864794*i_W1mv_a_r + 1.75782172888951*i_W1mv_b_i + 10.2390249864795*i_W1mv_b_r - 3.17407051442944*i_W1mv_c_i + 11.7109010138511*i_W1mv_c_r + 0.0982362237268616*i_W2lv_a_i - 0.0293174777213165*i_W2lv_a_r - 0.0982362237268642*i_W2lv_c_i + 0.0293174777213157*i_W2lv_c_r + 1.75765379075772*i_W2mv_a_i + 10.2388948546335*i_W2mv_a_r + 1.75765379075785*i_W2mv_b_i + 10.2388948546335*i_W2mv_b_r - 3.17423405384018*i_W2mv_c_i + 11.7107603897955*i_W2mv_c_r + 0.0982359608775129*i_W3lv_a_i - 0.0293168507523155*i_W3lv_a_r - 0.0982359608775155*i_W3lv_c_i + 0.0293168507523147*i_W3lv_c_r + 1.75714998466657*i_W3mv_a_i + 10.2385044573319*i_W3mv_a_r + 1.7571499846667*i_W3mv_b_i + 10.2385044573319*i_W3mv_b_r - 3.17472466374505*i_W3mv_c_i + 11.7103385159094*i_W3mv_c_r - 4.1812543389416e-5*v_GRID_a_i - 0.175092383128442*v_GRID_a_r + 3.1099782230896e-17*v_GRID_b_i - 4.08933647449996e-17*v_GRID_b_r + 4.18125433868462e-5*v_GRID_c_i + 0.175092383128446*v_GRID_c_r - v_W3mv_c_r
        struct[0].g[47,0] = -1.9180246741733*i_POI_a_i - 3.19497814474395*i_POI_a_r + 1.91802467417329*i_POI_c_i + 3.19497814474405*i_POI_c_r + 10.1951871226168*i_POImv_a_i - 1.79131033552721*i_POImv_a_r + 10.1951871226169*i_POImv_b_i - 1.79131033552733*i_POImv_b_r + 11.6029687203684*i_POImv_c_i + 3.08804231916218*i_POImv_c_r - 0.0293176867112787*i_W1lv_a_i - 0.098236311343155*i_W1lv_a_r + 0.0293176867112779*i_W1lv_c_i + 0.0982363113431576*i_W1lv_c_r + 10.2390249864794*i_W1mv_a_i - 1.75782172888938*i_W1mv_a_r + 10.2390249864795*i_W1mv_b_i - 1.75782172888951*i_W1mv_b_r + 11.7109010138511*i_W1mv_c_i + 3.17407051442944*i_W1mv_c_r - 0.0293174777213165*i_W2lv_a_i - 0.0982362237268616*i_W2lv_a_r + 0.0293174777213157*i_W2lv_c_i + 0.0982362237268642*i_W2lv_c_r + 10.2388948546335*i_W2mv_a_i - 1.75765379075772*i_W2mv_a_r + 10.2388948546335*i_W2mv_b_i - 1.75765379075785*i_W2mv_b_r + 11.7107603897955*i_W2mv_c_i + 3.17423405384018*i_W2mv_c_r - 0.0293168507523155*i_W3lv_a_i - 0.0982359608775129*i_W3lv_a_r + 0.0293168507523147*i_W3lv_c_i + 0.0982359608775155*i_W3lv_c_r + 10.2385044573319*i_W3mv_a_i - 1.75714998466657*i_W3mv_a_r + 10.2385044573319*i_W3mv_b_i - 1.7571499846667*i_W3mv_b_r + 11.7103385159094*i_W3mv_c_i + 3.17472466374505*i_W3mv_c_r - 0.175092383128442*v_GRID_a_i + 4.1812543389416e-5*v_GRID_a_r - 4.08933647449996e-17*v_GRID_b_i - 3.1099782230896e-17*v_GRID_b_r + 0.175092383128446*v_GRID_c_i - 4.18125433868462e-5*v_GRID_c_r - v_W3mv_c_i
        struct[0].g[48,0] = i_W1lv_a_i*v_W1lv_a_i + i_W1lv_a_r*v_W1lv_a_r - p_W1lv_a
        struct[0].g[49,0] = i_W1lv_b_i*v_W1lv_b_i + i_W1lv_b_r*v_W1lv_b_r - p_W1lv_b
        struct[0].g[50,0] = i_W1lv_c_i*v_W1lv_c_i + i_W1lv_c_r*v_W1lv_c_r - p_W1lv_c
        struct[0].g[51,0] = -i_W1lv_a_i*v_W1lv_a_r + i_W1lv_a_r*v_W1lv_a_i - q_W1lv_a
        struct[0].g[52,0] = -i_W1lv_b_i*v_W1lv_b_r + i_W1lv_b_r*v_W1lv_b_i - q_W1lv_b
        struct[0].g[53,0] = -i_W1lv_c_i*v_W1lv_c_r + i_W1lv_c_r*v_W1lv_c_i - q_W1lv_c
        struct[0].g[54,0] = i_W2lv_a_i*v_W2lv_a_i + i_W2lv_a_r*v_W2lv_a_r - p_W2lv_a
        struct[0].g[55,0] = i_W2lv_b_i*v_W2lv_b_i + i_W2lv_b_r*v_W2lv_b_r - p_W2lv_b
        struct[0].g[56,0] = i_W2lv_c_i*v_W2lv_c_i + i_W2lv_c_r*v_W2lv_c_r - p_W2lv_c
        struct[0].g[57,0] = -i_W2lv_a_i*v_W2lv_a_r + i_W2lv_a_r*v_W2lv_a_i - q_W2lv_a
        struct[0].g[58,0] = -i_W2lv_b_i*v_W2lv_b_r + i_W2lv_b_r*v_W2lv_b_i - q_W2lv_b
        struct[0].g[59,0] = -i_W2lv_c_i*v_W2lv_c_r + i_W2lv_c_r*v_W2lv_c_i - q_W2lv_c
        struct[0].g[60,0] = i_W3lv_a_i*v_W3lv_a_i + i_W3lv_a_r*v_W3lv_a_r - p_W3lv_a
        struct[0].g[61,0] = i_W3lv_b_i*v_W3lv_b_i + i_W3lv_b_r*v_W3lv_b_r - p_W3lv_b
        struct[0].g[62,0] = i_W3lv_c_i*v_W3lv_c_i + i_W3lv_c_r*v_W3lv_c_r - p_W3lv_c
        struct[0].g[63,0] = -i_W3lv_a_i*v_W3lv_a_r + i_W3lv_a_r*v_W3lv_a_i - q_W3lv_a
        struct[0].g[64,0] = -i_W3lv_b_i*v_W3lv_b_r + i_W3lv_b_r*v_W3lv_b_i - q_W3lv_b
        struct[0].g[65,0] = -i_W3lv_c_i*v_W3lv_c_r + i_W3lv_c_r*v_W3lv_c_i - q_W3lv_c
        struct[0].g[66,0] = i_POImv_a_i*v_POImv_a_i + i_POImv_a_r*v_POImv_a_r - p_POImv_a
        struct[0].g[67,0] = i_POImv_b_i*v_POImv_b_i + i_POImv_b_r*v_POImv_b_r - p_POImv_b
        struct[0].g[68,0] = i_POImv_c_i*v_POImv_c_i + i_POImv_c_r*v_POImv_c_r - p_POImv_c
        struct[0].g[69,0] = -i_POImv_a_i*v_POImv_a_r + i_POImv_a_r*v_POImv_a_i - q_POImv_a
        struct[0].g[70,0] = -i_POImv_b_i*v_POImv_b_r + i_POImv_b_r*v_POImv_b_i - q_POImv_b
        struct[0].g[71,0] = -i_POImv_c_i*v_POImv_c_r + i_POImv_c_r*v_POImv_c_i - q_POImv_c
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_GRID_a_i**2 + v_GRID_a_r**2)**0.5
        struct[0].h[1,0] = (v_GRID_b_i**2 + v_GRID_b_r**2)**0.5
        struct[0].h[2,0] = (v_GRID_c_i**2 + v_GRID_c_r**2)**0.5
        struct[0].h[3,0] = (v_W1lv_a_i**2 + v_W1lv_a_r**2)**0.5
        struct[0].h[4,0] = (v_W1lv_b_i**2 + v_W1lv_b_r**2)**0.5
        struct[0].h[5,0] = (v_W1lv_c_i**2 + v_W1lv_c_r**2)**0.5
        struct[0].h[6,0] = (v_W2lv_a_i**2 + v_W2lv_a_r**2)**0.5
        struct[0].h[7,0] = (v_W2lv_b_i**2 + v_W2lv_b_r**2)**0.5
        struct[0].h[8,0] = (v_W2lv_c_i**2 + v_W2lv_c_r**2)**0.5
        struct[0].h[9,0] = (v_W3lv_a_i**2 + v_W3lv_a_r**2)**0.5
        struct[0].h[10,0] = (v_W3lv_b_i**2 + v_W3lv_b_r**2)**0.5
        struct[0].h[11,0] = (v_W3lv_c_i**2 + v_W3lv_c_r**2)**0.5
        struct[0].h[12,0] = (v_POImv_a_i**2 + v_POImv_a_r**2)**0.5
        struct[0].h[13,0] = (v_POImv_b_i**2 + v_POImv_b_r**2)**0.5
        struct[0].h[14,0] = (v_POImv_c_i**2 + v_POImv_c_r**2)**0.5
        struct[0].h[15,0] = (v_POI_a_i**2 + v_POI_a_r**2)**0.5
        struct[0].h[16,0] = (v_POI_b_i**2 + v_POI_b_r**2)**0.5
        struct[0].h[17,0] = (v_POI_c_i**2 + v_POI_c_r**2)**0.5
        struct[0].h[18,0] = (v_W1mv_a_i**2 + v_W1mv_a_r**2)**0.5
        struct[0].h[19,0] = (v_W1mv_b_i**2 + v_W1mv_b_r**2)**0.5
        struct[0].h[20,0] = (v_W1mv_c_i**2 + v_W1mv_c_r**2)**0.5
        struct[0].h[21,0] = (v_W2mv_a_i**2 + v_W2mv_a_r**2)**0.5
        struct[0].h[22,0] = (v_W2mv_b_i**2 + v_W2mv_b_r**2)**0.5
        struct[0].h[23,0] = (v_W2mv_c_i**2 + v_W2mv_c_r**2)**0.5
        struct[0].h[24,0] = (v_W3mv_a_i**2 + v_W3mv_a_r**2)**0.5
        struct[0].h[25,0] = (v_W3mv_b_i**2 + v_W3mv_b_r**2)**0.5
        struct[0].h[26,0] = (v_W3mv_c_i**2 + v_W3mv_c_r**2)**0.5
    

    if mode == 10:

        struct[0].Fx[0,0] = -1

    if mode == 11:



        struct[0].Gy[0,0] = -1
        struct[0].Gy[0,48] = 0.00317393578459360
        struct[0].Gy[0,49] = -0.0154231877861473
        struct[0].Gy[0,50] = -0.000634767892296793
        struct[0].Gy[0,51] = 0.00199839389307364
        struct[0].Gy[0,52] = -0.000634767892296814
        struct[0].Gy[0,53] = 0.00199839389307368
        struct[0].Gy[0,54] = 0.00121874317417018
        struct[0].Gy[0,55] = -0.00395512560257793
        struct[0].Gy[0,56] = -0.000609371587085080
        struct[0].Gy[0,57] = 0.00197756280128894
        struct[0].Gy[0,58] = -0.000609371587085102
        struct[0].Gy[0,59] = 0.00197756280128899
        struct[0].Gy[0,60] = 0.00116793362771941
        struct[0].Gy[0,61] = -0.00391345649507333
        struct[0].Gy[0,62] = -0.000583966813859693
        struct[0].Gy[0,63] = 0.00195672824753664
        struct[0].Gy[0,64] = -0.000583966813859714
        struct[0].Gy[0,65] = 0.00195672824753669
        struct[0].Gy[0,66] = 0.0280418380652321
        struct[0].Gy[0,67] = -0.0971901504394946
        struct[0].Gy[0,70] = -0.0280418380652375
        struct[0].Gy[0,71] = 0.0971901504394895
        struct[0].Gy[1,1] = -1
        struct[0].Gy[1,48] = 0.0154231877861473
        struct[0].Gy[1,49] = 0.00317393578459360
        struct[0].Gy[1,50] = -0.00199839389307364
        struct[0].Gy[1,51] = -0.000634767892296793
        struct[0].Gy[1,52] = -0.00199839389307368
        struct[0].Gy[1,53] = -0.000634767892296814
        struct[0].Gy[1,54] = 0.00395512560257793
        struct[0].Gy[1,55] = 0.00121874317417018
        struct[0].Gy[1,56] = -0.00197756280128894
        struct[0].Gy[1,57] = -0.000609371587085080
        struct[0].Gy[1,58] = -0.00197756280128899
        struct[0].Gy[1,59] = -0.000609371587085102
        struct[0].Gy[1,60] = 0.00391345649507333
        struct[0].Gy[1,61] = 0.00116793362771941
        struct[0].Gy[1,62] = -0.00195672824753664
        struct[0].Gy[1,63] = -0.000583966813859693
        struct[0].Gy[1,64] = -0.00195672824753669
        struct[0].Gy[1,65] = -0.000583966813859714
        struct[0].Gy[1,66] = 0.0971901504394946
        struct[0].Gy[1,67] = 0.0280418380652321
        struct[0].Gy[1,70] = -0.0971901504394895
        struct[0].Gy[1,71] = -0.0280418380652375
        struct[0].Gy[2,2] = -1
        struct[0].Gy[2,48] = -0.000634767892296777
        struct[0].Gy[2,49] = 0.00199839389307366
        struct[0].Gy[2,50] = 0.00317393578459362
        struct[0].Gy[2,51] = -0.0154231877861474
        struct[0].Gy[2,52] = -0.000634767892296846
        struct[0].Gy[2,53] = 0.00199839389307372
        struct[0].Gy[2,54] = -0.000609371587085066
        struct[0].Gy[2,55] = 0.00197756280128896
        struct[0].Gy[2,56] = 0.00121874317417020
        struct[0].Gy[2,57] = -0.00395512560257798
        struct[0].Gy[2,58] = -0.000609371587085134
        struct[0].Gy[2,59] = 0.00197756280128902
        struct[0].Gy[2,60] = -0.000583966813859679
        struct[0].Gy[2,61] = 0.00195672824753666
        struct[0].Gy[2,62] = 0.00116793362771943
        struct[0].Gy[2,63] = -0.00391345649507338
        struct[0].Gy[2,64] = -0.000583966813859747
        struct[0].Gy[2,65] = 0.00195672824753672
        struct[0].Gy[2,66] = -0.0280418380652299
        struct[0].Gy[2,67] = 0.0971901504394967
        struct[0].Gy[2,68] = 0.0280418380652406
        struct[0].Gy[2,69] = -0.0971901504394895
        struct[0].Gy[3,3] = -1
        struct[0].Gy[3,48] = -0.00199839389307366
        struct[0].Gy[3,49] = -0.000634767892296777
        struct[0].Gy[3,50] = 0.0154231877861474
        struct[0].Gy[3,51] = 0.00317393578459362
        struct[0].Gy[3,52] = -0.00199839389307372
        struct[0].Gy[3,53] = -0.000634767892296846
        struct[0].Gy[3,54] = -0.00197756280128896
        struct[0].Gy[3,55] = -0.000609371587085066
        struct[0].Gy[3,56] = 0.00395512560257798
        struct[0].Gy[3,57] = 0.00121874317417020
        struct[0].Gy[3,58] = -0.00197756280128902
        struct[0].Gy[3,59] = -0.000609371587085134
        struct[0].Gy[3,60] = -0.00195672824753666
        struct[0].Gy[3,61] = -0.000583966813859679
        struct[0].Gy[3,62] = 0.00391345649507338
        struct[0].Gy[3,63] = 0.00116793362771943
        struct[0].Gy[3,64] = -0.00195672824753672
        struct[0].Gy[3,65] = -0.000583966813859747
        struct[0].Gy[3,66] = -0.0971901504394967
        struct[0].Gy[3,67] = -0.0280418380652299
        struct[0].Gy[3,68] = 0.0971901504394895
        struct[0].Gy[3,69] = 0.0280418380652406
        struct[0].Gy[4,4] = -1
        struct[0].Gy[4,48] = -0.000634767892296827
        struct[0].Gy[4,49] = 0.00199839389307367
        struct[0].Gy[4,50] = -0.000634767892296831
        struct[0].Gy[4,51] = 0.00199839389307374
        struct[0].Gy[4,52] = 0.00317393578459366
        struct[0].Gy[4,53] = -0.0154231877861474
        struct[0].Gy[4,54] = -0.000609371587085114
        struct[0].Gy[4,55] = 0.00197756280128897
        struct[0].Gy[4,56] = -0.000609371587085120
        struct[0].Gy[4,57] = 0.00197756280128904
        struct[0].Gy[4,58] = 0.00121874317417024
        struct[0].Gy[4,59] = -0.00395512560257801
        struct[0].Gy[4,60] = -0.000583966813859728
        struct[0].Gy[4,61] = 0.00195672824753667
        struct[0].Gy[4,62] = -0.000583966813859733
        struct[0].Gy[4,63] = 0.00195672824753674
        struct[0].Gy[4,64] = 0.00116793362771946
        struct[0].Gy[4,65] = -0.00391345649507341
        struct[0].Gy[4,68] = -0.0280418380652387
        struct[0].Gy[4,69] = 0.0971901504394933
        struct[0].Gy[4,70] = 0.0280418380652339
        struct[0].Gy[4,71] = -0.0971901504394944
        struct[0].Gy[5,5] = -1
        struct[0].Gy[5,48] = -0.00199839389307367
        struct[0].Gy[5,49] = -0.000634767892296827
        struct[0].Gy[5,50] = -0.00199839389307374
        struct[0].Gy[5,51] = -0.000634767892296831
        struct[0].Gy[5,52] = 0.0154231877861474
        struct[0].Gy[5,53] = 0.00317393578459366
        struct[0].Gy[5,54] = -0.00197756280128897
        struct[0].Gy[5,55] = -0.000609371587085114
        struct[0].Gy[5,56] = -0.00197756280128904
        struct[0].Gy[5,57] = -0.000609371587085120
        struct[0].Gy[5,58] = 0.00395512560257801
        struct[0].Gy[5,59] = 0.00121874317417024
        struct[0].Gy[5,60] = -0.00195672824753667
        struct[0].Gy[5,61] = -0.000583966813859728
        struct[0].Gy[5,62] = -0.00195672824753674
        struct[0].Gy[5,63] = -0.000583966813859733
        struct[0].Gy[5,64] = 0.00391345649507341
        struct[0].Gy[5,65] = 0.00116793362771946
        struct[0].Gy[5,68] = -0.0971901504394933
        struct[0].Gy[5,69] = -0.0280418380652387
        struct[0].Gy[5,70] = 0.0971901504394944
        struct[0].Gy[5,71] = 0.0280418380652339
        struct[0].Gy[6,6] = -1
        struct[0].Gy[6,48] = 0.00121874317417018
        struct[0].Gy[6,49] = -0.00395512560257793
        struct[0].Gy[6,50] = -0.000609371587085081
        struct[0].Gy[6,51] = 0.00197756280128894
        struct[0].Gy[6,52] = -0.000609371587085101
        struct[0].Gy[6,53] = 0.00197756280128899
        struct[0].Gy[6,54] = 0.00312313470615651
        struct[0].Gy[6,55] = -0.0153815221406103
        struct[0].Gy[6,56] = -0.000609367353078243
        struct[0].Gy[6,57] = 0.00197756107030514
        struct[0].Gy[6,58] = -0.000609367353078263
        struct[0].Gy[6,59] = 0.00197756107030519
        struct[0].Gy[6,60] = 0.00116792530215105
        struct[0].Gy[6,61] = -0.00391345300468828
        struct[0].Gy[6,62] = -0.000583962651075517
        struct[0].Gy[6,63] = 0.00195672650234412
        struct[0].Gy[6,64] = -0.000583962651075537
        struct[0].Gy[6,65] = 0.00195672650234417
        struct[0].Gy[6,66] = 0.0280416326518444
        struct[0].Gy[6,67] = -0.0971900621093923
        struct[0].Gy[6,70] = -0.0280416326518497
        struct[0].Gy[6,71] = 0.0971900621093875
        struct[0].Gy[7,7] = -1
        struct[0].Gy[7,48] = 0.00395512560257793
        struct[0].Gy[7,49] = 0.00121874317417018
        struct[0].Gy[7,50] = -0.00197756280128894
        struct[0].Gy[7,51] = -0.000609371587085081
        struct[0].Gy[7,52] = -0.00197756280128899
        struct[0].Gy[7,53] = -0.000609371587085101
        struct[0].Gy[7,54] = 0.0153815221406103
        struct[0].Gy[7,55] = 0.00312313470615651
        struct[0].Gy[7,56] = -0.00197756107030514
        struct[0].Gy[7,57] = -0.000609367353078243
        struct[0].Gy[7,58] = -0.00197756107030519
        struct[0].Gy[7,59] = -0.000609367353078263
        struct[0].Gy[7,60] = 0.00391345300468828
        struct[0].Gy[7,61] = 0.00116792530215105
        struct[0].Gy[7,62] = -0.00195672650234412
        struct[0].Gy[7,63] = -0.000583962651075517
        struct[0].Gy[7,64] = -0.00195672650234417
        struct[0].Gy[7,65] = -0.000583962651075537
        struct[0].Gy[7,66] = 0.0971900621093923
        struct[0].Gy[7,67] = 0.0280416326518444
        struct[0].Gy[7,70] = -0.0971900621093875
        struct[0].Gy[7,71] = -0.0280416326518497
        struct[0].Gy[8,8] = -1
        struct[0].Gy[8,48] = -0.000609371587085065
        struct[0].Gy[8,49] = 0.00197756280128896
        struct[0].Gy[8,50] = 0.00121874317417020
        struct[0].Gy[8,51] = -0.00395512560257798
        struct[0].Gy[8,52] = -0.000609371587085133
        struct[0].Gy[8,53] = 0.00197756280128902
        struct[0].Gy[8,54] = -0.000609367353078228
        struct[0].Gy[8,55] = 0.00197756107030516
        struct[0].Gy[8,56] = 0.00312313470615652
        struct[0].Gy[8,57] = -0.0153815221406104
        struct[0].Gy[8,58] = -0.000609367353078295
        struct[0].Gy[8,59] = 0.00197756107030522
        struct[0].Gy[8,60] = -0.000583962651075503
        struct[0].Gy[8,61] = 0.00195672650234414
        struct[0].Gy[8,62] = 0.00116792530215107
        struct[0].Gy[8,63] = -0.00391345300468834
        struct[0].Gy[8,64] = -0.000583962651075570
        struct[0].Gy[8,65] = 0.00195672650234420
        struct[0].Gy[8,66] = -0.0280416326518423
        struct[0].Gy[8,67] = 0.0971900621093946
        struct[0].Gy[8,68] = 0.0280416326518527
        struct[0].Gy[8,69] = -0.0971900621093876
        struct[0].Gy[9,9] = -1
        struct[0].Gy[9,48] = -0.00197756280128896
        struct[0].Gy[9,49] = -0.000609371587085065
        struct[0].Gy[9,50] = 0.00395512560257798
        struct[0].Gy[9,51] = 0.00121874317417020
        struct[0].Gy[9,52] = -0.00197756280128902
        struct[0].Gy[9,53] = -0.000609371587085133
        struct[0].Gy[9,54] = -0.00197756107030516
        struct[0].Gy[9,55] = -0.000609367353078228
        struct[0].Gy[9,56] = 0.0153815221406104
        struct[0].Gy[9,57] = 0.00312313470615652
        struct[0].Gy[9,58] = -0.00197756107030522
        struct[0].Gy[9,59] = -0.000609367353078295
        struct[0].Gy[9,60] = -0.00195672650234414
        struct[0].Gy[9,61] = -0.000583962651075503
        struct[0].Gy[9,62] = 0.00391345300468834
        struct[0].Gy[9,63] = 0.00116792530215107
        struct[0].Gy[9,64] = -0.00195672650234420
        struct[0].Gy[9,65] = -0.000583962651075570
        struct[0].Gy[9,66] = -0.0971900621093946
        struct[0].Gy[9,67] = -0.0280416326518423
        struct[0].Gy[9,68] = 0.0971900621093876
        struct[0].Gy[9,69] = 0.0280416326518527
        struct[0].Gy[10,10] = -1
        struct[0].Gy[10,48] = -0.000609371587085115
        struct[0].Gy[10,49] = 0.00197756280128897
        struct[0].Gy[10,50] = -0.000609371587085118
        struct[0].Gy[10,51] = 0.00197756280128904
        struct[0].Gy[10,52] = 0.00121874317417023
        struct[0].Gy[10,53] = -0.00395512560257801
        struct[0].Gy[10,54] = -0.000609367353078276
        struct[0].Gy[10,55] = 0.00197756107030517
        struct[0].Gy[10,56] = -0.000609367353078280
        struct[0].Gy[10,57] = 0.00197756107030524
        struct[0].Gy[10,58] = 0.00312313470615656
        struct[0].Gy[10,59] = -0.0153815221406104
        struct[0].Gy[10,60] = -0.000583962651075551
        struct[0].Gy[10,61] = 0.00195672650234415
        struct[0].Gy[10,62] = -0.000583962651075555
        struct[0].Gy[10,63] = 0.00195672650234422
        struct[0].Gy[10,64] = 0.00116792530215111
        struct[0].Gy[10,65] = -0.00391345300468836
        struct[0].Gy[10,68] = -0.0280416326518508
        struct[0].Gy[10,69] = 0.0971900621093912
        struct[0].Gy[10,70] = 0.0280416326518462
        struct[0].Gy[10,71] = -0.0971900621093924
        struct[0].Gy[11,11] = -1
        struct[0].Gy[11,48] = -0.00197756280128897
        struct[0].Gy[11,49] = -0.000609371587085115
        struct[0].Gy[11,50] = -0.00197756280128904
        struct[0].Gy[11,51] = -0.000609371587085118
        struct[0].Gy[11,52] = 0.00395512560257801
        struct[0].Gy[11,53] = 0.00121874317417023
        struct[0].Gy[11,54] = -0.00197756107030517
        struct[0].Gy[11,55] = -0.000609367353078276
        struct[0].Gy[11,56] = -0.00197756107030524
        struct[0].Gy[11,57] = -0.000609367353078280
        struct[0].Gy[11,58] = 0.0153815221406104
        struct[0].Gy[11,59] = 0.00312313470615656
        struct[0].Gy[11,60] = -0.00195672650234415
        struct[0].Gy[11,61] = -0.000583962651075551
        struct[0].Gy[11,62] = -0.00195672650234422
        struct[0].Gy[11,63] = -0.000583962651075555
        struct[0].Gy[11,64] = 0.00391345300468836
        struct[0].Gy[11,65] = 0.00116792530215111
        struct[0].Gy[11,68] = -0.0971900621093912
        struct[0].Gy[11,69] = -0.0280416326518508
        struct[0].Gy[11,70] = 0.0971900621093924
        struct[0].Gy[11,71] = 0.0280416326518462
        struct[0].Gy[12,12] = -1
        struct[0].Gy[12,48] = 0.00116793362771941
        struct[0].Gy[12,49] = -0.00391345649507333
        struct[0].Gy[12,50] = -0.000583966813859694
        struct[0].Gy[12,51] = 0.00195672824753664
        struct[0].Gy[12,52] = -0.000583966813859713
        struct[0].Gy[12,53] = 0.00195672824753669
        struct[0].Gy[12,54] = 0.00116792530215105
        struct[0].Gy[12,55] = -0.00391345300468828
        struct[0].Gy[12,56] = -0.000583962651075518
        struct[0].Gy[12,57] = 0.00195672650234412
        struct[0].Gy[12,58] = -0.000583962651075537
        struct[0].Gy[12,59] = 0.00195672650234417
        struct[0].Gy[12,60] = 0.00307230032548127
        struct[0].Gy[12,61] = -0.0153398425335145
        struct[0].Gy[12,62] = -0.000583950162740626
        struct[0].Gy[12,63] = 0.00195672126675721
        struct[0].Gy[12,64] = -0.000583950162740645
        struct[0].Gy[12,65] = 0.00195672126675726
        struct[0].Gy[12,66] = 0.0280410164125591
        struct[0].Gy[12,67] = -0.0971897971186317
        struct[0].Gy[12,70] = -0.0280410164125642
        struct[0].Gy[12,71] = 0.0971897971186269
        struct[0].Gy[13,13] = -1
        struct[0].Gy[13,48] = 0.00391345649507333
        struct[0].Gy[13,49] = 0.00116793362771941
        struct[0].Gy[13,50] = -0.00195672824753664
        struct[0].Gy[13,51] = -0.000583966813859694
        struct[0].Gy[13,52] = -0.00195672824753669
        struct[0].Gy[13,53] = -0.000583966813859713
        struct[0].Gy[13,54] = 0.00391345300468828
        struct[0].Gy[13,55] = 0.00116792530215105
        struct[0].Gy[13,56] = -0.00195672650234412
        struct[0].Gy[13,57] = -0.000583962651075518
        struct[0].Gy[13,58] = -0.00195672650234417
        struct[0].Gy[13,59] = -0.000583962651075537
        struct[0].Gy[13,60] = 0.0153398425335145
        struct[0].Gy[13,61] = 0.00307230032548127
        struct[0].Gy[13,62] = -0.00195672126675721
        struct[0].Gy[13,63] = -0.000583950162740626
        struct[0].Gy[13,64] = -0.00195672126675726
        struct[0].Gy[13,65] = -0.000583950162740645
        struct[0].Gy[13,66] = 0.0971897971186317
        struct[0].Gy[13,67] = 0.0280410164125591
        struct[0].Gy[13,70] = -0.0971897971186269
        struct[0].Gy[13,71] = -0.0280410164125642
        struct[0].Gy[14,14] = -1
        struct[0].Gy[14,48] = -0.000583966813859680
        struct[0].Gy[14,49] = 0.00195672824753666
        struct[0].Gy[14,50] = 0.00116793362771943
        struct[0].Gy[14,51] = -0.00391345649507338
        struct[0].Gy[14,52] = -0.000583966813859745
        struct[0].Gy[14,53] = 0.00195672824753672
        struct[0].Gy[14,54] = -0.000583962651075503
        struct[0].Gy[14,55] = 0.00195672650234414
        struct[0].Gy[14,56] = 0.00116792530215107
        struct[0].Gy[14,57] = -0.00391345300468834
        struct[0].Gy[14,58] = -0.000583962651075569
        struct[0].Gy[14,59] = 0.00195672650234420
        struct[0].Gy[14,60] = -0.000583950162740612
        struct[0].Gy[14,61] = 0.00195672126675723
        struct[0].Gy[14,62] = 0.00307230032548129
        struct[0].Gy[14,63] = -0.0153398425335145
        struct[0].Gy[14,64] = -0.000583950162740677
        struct[0].Gy[14,65] = 0.00195672126675729
        struct[0].Gy[14,66] = -0.0280410164125570
        struct[0].Gy[14,67] = 0.0971897971186340
        struct[0].Gy[14,68] = 0.0280410164125671
        struct[0].Gy[14,69] = -0.0971897971186271
        struct[0].Gy[15,15] = -1
        struct[0].Gy[15,48] = -0.00195672824753666
        struct[0].Gy[15,49] = -0.000583966813859680
        struct[0].Gy[15,50] = 0.00391345649507338
        struct[0].Gy[15,51] = 0.00116793362771943
        struct[0].Gy[15,52] = -0.00195672824753672
        struct[0].Gy[15,53] = -0.000583966813859745
        struct[0].Gy[15,54] = -0.00195672650234414
        struct[0].Gy[15,55] = -0.000583962651075503
        struct[0].Gy[15,56] = 0.00391345300468834
        struct[0].Gy[15,57] = 0.00116792530215107
        struct[0].Gy[15,58] = -0.00195672650234420
        struct[0].Gy[15,59] = -0.000583962651075569
        struct[0].Gy[15,60] = -0.00195672126675723
        struct[0].Gy[15,61] = -0.000583950162740612
        struct[0].Gy[15,62] = 0.0153398425335145
        struct[0].Gy[15,63] = 0.00307230032548129
        struct[0].Gy[15,64] = -0.00195672126675729
        struct[0].Gy[15,65] = -0.000583950162740677
        struct[0].Gy[15,66] = -0.0971897971186340
        struct[0].Gy[15,67] = -0.0280410164125570
        struct[0].Gy[15,68] = 0.0971897971186271
        struct[0].Gy[15,69] = 0.0280410164125671
        struct[0].Gy[16,16] = -1
        struct[0].Gy[16,48] = -0.000583966813859728
        struct[0].Gy[16,49] = 0.00195672824753667
        struct[0].Gy[16,50] = -0.000583966813859731
        struct[0].Gy[16,51] = 0.00195672824753674
        struct[0].Gy[16,52] = 0.00116793362771946
        struct[0].Gy[16,53] = -0.00391345649507341
        struct[0].Gy[16,54] = -0.000583962651075551
        struct[0].Gy[16,55] = 0.00195672650234415
        struct[0].Gy[16,56] = -0.000583962651075554
        struct[0].Gy[16,57] = 0.00195672650234422
        struct[0].Gy[16,58] = 0.00116792530215111
        struct[0].Gy[16,59] = -0.00391345300468836
        struct[0].Gy[16,60] = -0.000583950162740659
        struct[0].Gy[16,61] = 0.00195672126675724
        struct[0].Gy[16,62] = -0.000583950162740663
        struct[0].Gy[16,63] = 0.00195672126675731
        struct[0].Gy[16,64] = 0.00307230032548132
        struct[0].Gy[16,65] = -0.0153398425335145
        struct[0].Gy[16,68] = -0.0280410164125652
        struct[0].Gy[16,69] = 0.0971897971186306
        struct[0].Gy[16,70] = 0.0280410164125607
        struct[0].Gy[16,71] = -0.0971897971186319
        struct[0].Gy[17,17] = -1
        struct[0].Gy[17,48] = -0.00195672824753667
        struct[0].Gy[17,49] = -0.000583966813859728
        struct[0].Gy[17,50] = -0.00195672824753674
        struct[0].Gy[17,51] = -0.000583966813859731
        struct[0].Gy[17,52] = 0.00391345649507341
        struct[0].Gy[17,53] = 0.00116793362771946
        struct[0].Gy[17,54] = -0.00195672650234415
        struct[0].Gy[17,55] = -0.000583962651075551
        struct[0].Gy[17,56] = -0.00195672650234422
        struct[0].Gy[17,57] = -0.000583962651075554
        struct[0].Gy[17,58] = 0.00391345300468836
        struct[0].Gy[17,59] = 0.00116792530215111
        struct[0].Gy[17,60] = -0.00195672126675724
        struct[0].Gy[17,61] = -0.000583950162740659
        struct[0].Gy[17,62] = -0.00195672126675731
        struct[0].Gy[17,63] = -0.000583950162740663
        struct[0].Gy[17,64] = 0.0153398425335145
        struct[0].Gy[17,65] = 0.00307230032548132
        struct[0].Gy[17,68] = -0.0971897971186306
        struct[0].Gy[17,69] = -0.0280410164125652
        struct[0].Gy[17,70] = 0.0971897971186319
        struct[0].Gy[17,71] = 0.0280410164125607
        struct[0].Gy[18,18] = -1
        struct[0].Gy[18,48] = 0.0280418380652350
        struct[0].Gy[18,49] = -0.0971901504394881
        struct[0].Gy[18,50] = -0.0280418380652335
        struct[0].Gy[18,51] = 0.0971901504394933
        struct[0].Gy[18,54] = 0.0280416326518473
        struct[0].Gy[18,55] = -0.0971900621093861
        struct[0].Gy[18,56] = -0.0280416326518457
        struct[0].Gy[18,57] = 0.0971900621093912
        struct[0].Gy[18,60] = 0.0280410164125620
        struct[0].Gy[18,61] = -0.0971897971186256
        struct[0].Gy[18,62] = -0.0280410164125604
        struct[0].Gy[18,63] = 0.0971897971186307
        struct[0].Gy[18,66] = 11.6022742434667
        struct[0].Gy[18,67] = -3.08885814101954
        struct[0].Gy[18,68] = 10.1945442087447
        struct[0].Gy[18,69] = 1.79047234076949
        struct[0].Gy[18,70] = 10.1945442087447
        struct[0].Gy[18,71] = 1.79047234076923
        struct[0].Gy[19,19] = -1
        struct[0].Gy[19,48] = 0.0971901504394881
        struct[0].Gy[19,49] = 0.0280418380652350
        struct[0].Gy[19,50] = -0.0971901504394933
        struct[0].Gy[19,51] = -0.0280418380652335
        struct[0].Gy[19,54] = 0.0971900621093861
        struct[0].Gy[19,55] = 0.0280416326518473
        struct[0].Gy[19,56] = -0.0971900621093912
        struct[0].Gy[19,57] = -0.0280416326518457
        struct[0].Gy[19,60] = 0.0971897971186256
        struct[0].Gy[19,61] = 0.0280410164125620
        struct[0].Gy[19,62] = -0.0971897971186307
        struct[0].Gy[19,63] = -0.0280410164125604
        struct[0].Gy[19,66] = 3.08885814101954
        struct[0].Gy[19,67] = 11.6022742434667
        struct[0].Gy[19,68] = -1.79047234076949
        struct[0].Gy[19,69] = 10.1945442087447
        struct[0].Gy[19,70] = -1.79047234076923
        struct[0].Gy[19,71] = 10.1945442087447
        struct[0].Gy[20,20] = -1
        struct[0].Gy[20,50] = 0.0280418380652370
        struct[0].Gy[20,51] = -0.0971901504394930
        struct[0].Gy[20,52] = -0.0280418380652384
        struct[0].Gy[20,53] = 0.0971901504394895
        struct[0].Gy[20,56] = 0.0280416326518492
        struct[0].Gy[20,57] = -0.0971900621093910
        struct[0].Gy[20,58] = -0.0280416326518506
        struct[0].Gy[20,59] = 0.0971900621093875
        struct[0].Gy[20,62] = 0.0280410164125637
        struct[0].Gy[20,63] = -0.0971897971186304
        struct[0].Gy[20,64] = -0.0280410164125652
        struct[0].Gy[20,65] = 0.0971897971186269
        struct[0].Gy[20,66] = 10.1945442087449
        struct[0].Gy[20,67] = 1.79047234076965
        struct[0].Gy[20,68] = 11.6022742434671
        struct[0].Gy[20,69] = -3.08885814101935
        struct[0].Gy[20,70] = 10.1945442087448
        struct[0].Gy[20,71] = 1.79047234076948
        struct[0].Gy[21,21] = -1
        struct[0].Gy[21,50] = 0.0971901504394930
        struct[0].Gy[21,51] = 0.0280418380652370
        struct[0].Gy[21,52] = -0.0971901504394895
        struct[0].Gy[21,53] = -0.0280418380652384
        struct[0].Gy[21,56] = 0.0971900621093910
        struct[0].Gy[21,57] = 0.0280416326518492
        struct[0].Gy[21,58] = -0.0971900621093875
        struct[0].Gy[21,59] = -0.0280416326518506
        struct[0].Gy[21,62] = 0.0971897971186304
        struct[0].Gy[21,63] = 0.0280410164125637
        struct[0].Gy[21,64] = -0.0971897971186269
        struct[0].Gy[21,65] = -0.0280410164125652
        struct[0].Gy[21,66] = -1.79047234076965
        struct[0].Gy[21,67] = 10.1945442087449
        struct[0].Gy[21,68] = 3.08885814101935
        struct[0].Gy[21,69] = 11.6022742434671
        struct[0].Gy[21,70] = -1.79047234076948
        struct[0].Gy[21,71] = 10.1945442087448
        struct[0].Gy[22,22] = -1
        struct[0].Gy[22,48] = -0.0280418380652345
        struct[0].Gy[22,49] = 0.0971901504394956
        struct[0].Gy[22,52] = 0.0280418380652337
        struct[0].Gy[22,53] = -0.0971901504394982
        struct[0].Gy[22,54] = -0.0280416326518468
        struct[0].Gy[22,55] = 0.0971900621093936
        struct[0].Gy[22,58] = 0.0280416326518460
        struct[0].Gy[22,59] = -0.0971900621093962
        struct[0].Gy[22,60] = -0.0280410164125613
        struct[0].Gy[22,61] = 0.0971897971186331
        struct[0].Gy[22,64] = 0.0280410164125606
        struct[0].Gy[22,65] = -0.0971897971186357
        struct[0].Gy[22,66] = 10.1945442087448
        struct[0].Gy[22,67] = 1.79047234076953
        struct[0].Gy[22,68] = 10.1945442087448
        struct[0].Gy[22,69] = 1.79047234076966
        struct[0].Gy[22,70] = 11.6022742434669
        struct[0].Gy[22,71] = -3.08885814101960
        struct[0].Gy[23,23] = -1
        struct[0].Gy[23,48] = -0.0971901504394956
        struct[0].Gy[23,49] = -0.0280418380652345
        struct[0].Gy[23,52] = 0.0971901504394982
        struct[0].Gy[23,53] = 0.0280418380652337
        struct[0].Gy[23,54] = -0.0971900621093936
        struct[0].Gy[23,55] = -0.0280416326518468
        struct[0].Gy[23,58] = 0.0971900621093962
        struct[0].Gy[23,59] = 0.0280416326518460
        struct[0].Gy[23,60] = -0.0971897971186331
        struct[0].Gy[23,61] = -0.0280410164125613
        struct[0].Gy[23,64] = 0.0971897971186357
        struct[0].Gy[23,65] = 0.0280410164125606
        struct[0].Gy[23,66] = -1.79047234076953
        struct[0].Gy[23,67] = 10.1945442087448
        struct[0].Gy[23,68] = -1.79047234076966
        struct[0].Gy[23,69] = 10.1945442087448
        struct[0].Gy[23,70] = 3.08885814101960
        struct[0].Gy[23,71] = 11.6022742434669
        struct[0].Gy[24,24] = -1
        struct[0].Gy[24,48] = 0.0764099708538850
        struct[0].Gy[24,49] = -0.127279074345343
        struct[0].Gy[24,50] = -0.0382049854269419
        struct[0].Gy[24,51] = 0.0636395371726706
        struct[0].Gy[24,52] = -0.0382049854269430
        struct[0].Gy[24,53] = 0.0636395371726721
        struct[0].Gy[24,54] = 0.0764096462087186
        struct[0].Gy[24,55] = -0.127279026494919
        struct[0].Gy[24,56] = -0.0382048231043588
        struct[0].Gy[24,57] = 0.0636395132474590
        struct[0].Gy[24,58] = -0.0382048231043599
        struct[0].Gy[24,59] = 0.0636395132474605
        struct[0].Gy[24,60] = 0.0764086722742935
        struct[0].Gy[24,61] = -0.127278882942674
        struct[0].Gy[24,62] = -0.0382043361371462
        struct[0].Gy[24,63] = 0.0636394414713363
        struct[0].Gy[24,64] = -0.0382043361371473
        struct[0].Gy[24,65] = 0.0636394414713379
        struct[0].Gy[24,66] = 1.91798392779186
        struct[0].Gy[24,67] = -3.19497213887046
        struct[0].Gy[24,70] = -1.91798392779199
        struct[0].Gy[24,71] = 3.19497213887024
        struct[0].Gy[25,25] = -1
        struct[0].Gy[25,48] = 0.127279074345343
        struct[0].Gy[25,49] = 0.0764099708538850
        struct[0].Gy[25,50] = -0.0636395371726706
        struct[0].Gy[25,51] = -0.0382049854269419
        struct[0].Gy[25,52] = -0.0636395371726721
        struct[0].Gy[25,53] = -0.0382049854269430
        struct[0].Gy[25,54] = 0.127279026494919
        struct[0].Gy[25,55] = 0.0764096462087186
        struct[0].Gy[25,56] = -0.0636395132474590
        struct[0].Gy[25,57] = -0.0382048231043588
        struct[0].Gy[25,58] = -0.0636395132474605
        struct[0].Gy[25,59] = -0.0382048231043599
        struct[0].Gy[25,60] = 0.127278882942674
        struct[0].Gy[25,61] = 0.0764086722742935
        struct[0].Gy[25,62] = -0.0636394414713363
        struct[0].Gy[25,63] = -0.0382043361371462
        struct[0].Gy[25,64] = -0.0636394414713379
        struct[0].Gy[25,65] = -0.0382043361371473
        struct[0].Gy[25,66] = 3.19497213887046
        struct[0].Gy[25,67] = 1.91798392779186
        struct[0].Gy[25,70] = -3.19497213887024
        struct[0].Gy[25,71] = -1.91798392779199
        struct[0].Gy[26,26] = -1
        struct[0].Gy[26,48] = -0.0382049854269416
        struct[0].Gy[26,49] = 0.0636395371726714
        struct[0].Gy[26,50] = 0.0764099708538861
        struct[0].Gy[26,51] = -0.127279074345344
        struct[0].Gy[26,52] = -0.0382049854269445
        struct[0].Gy[26,53] = 0.0636395371726729
        struct[0].Gy[26,54] = -0.0382048231043584
        struct[0].Gy[26,55] = 0.0636395132474598
        struct[0].Gy[26,56] = 0.0764096462087197
        struct[0].Gy[26,57] = -0.127279026494921
        struct[0].Gy[26,58] = -0.0382048231043613
        struct[0].Gy[26,59] = 0.0636395132474612
        struct[0].Gy[26,60] = -0.0382043361371459
        struct[0].Gy[26,61] = 0.0636394414713371
        struct[0].Gy[26,62] = 0.0764086722742946
        struct[0].Gy[26,63] = -0.127278882942676
        struct[0].Gy[26,64] = -0.0382043361371488
        struct[0].Gy[26,65] = 0.0636394414713386
        struct[0].Gy[26,66] = -1.91798392779181
        struct[0].Gy[26,67] = 3.19497213887056
        struct[0].Gy[26,68] = 1.91798392779210
        struct[0].Gy[26,69] = -3.19497213887022
        struct[0].Gy[27,27] = -1
        struct[0].Gy[27,48] = -0.0636395371726714
        struct[0].Gy[27,49] = -0.0382049854269416
        struct[0].Gy[27,50] = 0.127279074345344
        struct[0].Gy[27,51] = 0.0764099708538861
        struct[0].Gy[27,52] = -0.0636395371726729
        struct[0].Gy[27,53] = -0.0382049854269445
        struct[0].Gy[27,54] = -0.0636395132474598
        struct[0].Gy[27,55] = -0.0382048231043584
        struct[0].Gy[27,56] = 0.127279026494921
        struct[0].Gy[27,57] = 0.0764096462087197
        struct[0].Gy[27,58] = -0.0636395132474612
        struct[0].Gy[27,59] = -0.0382048231043613
        struct[0].Gy[27,60] = -0.0636394414713371
        struct[0].Gy[27,61] = -0.0382043361371459
        struct[0].Gy[27,62] = 0.127278882942676
        struct[0].Gy[27,63] = 0.0764086722742946
        struct[0].Gy[27,64] = -0.0636394414713386
        struct[0].Gy[27,65] = -0.0382043361371488
        struct[0].Gy[27,66] = -3.19497213887056
        struct[0].Gy[27,67] = -1.91798392779181
        struct[0].Gy[27,68] = 3.19497213887022
        struct[0].Gy[27,69] = 1.91798392779210
        struct[0].Gy[28,28] = -1
        struct[0].Gy[28,48] = -0.0382049854269434
        struct[0].Gy[28,49] = 0.0636395371726713
        struct[0].Gy[28,50] = -0.0382049854269442
        struct[0].Gy[28,51] = 0.0636395371726737
        struct[0].Gy[28,52] = 0.0764099708538875
        struct[0].Gy[28,53] = -0.127279074345345
        struct[0].Gy[28,54] = -0.0382048231043602
        struct[0].Gy[28,55] = 0.0636395132474596
        struct[0].Gy[28,56] = -0.0382048231043610
        struct[0].Gy[28,57] = 0.0636395132474621
        struct[0].Gy[28,58] = 0.0764096462087212
        struct[0].Gy[28,59] = -0.127279026494922
        struct[0].Gy[28,60] = -0.0382043361371476
        struct[0].Gy[28,61] = 0.0636394414713370
        struct[0].Gy[28,62] = -0.0382043361371484
        struct[0].Gy[28,63] = 0.0636394414713394
        struct[0].Gy[28,64] = 0.0764086722742961
        struct[0].Gy[28,65] = -0.127278882942676
        struct[0].Gy[28,68] = -1.91798392779206
        struct[0].Gy[28,69] = 3.19497213887036
        struct[0].Gy[28,70] = 1.91798392779192
        struct[0].Gy[28,71] = -3.19497213887045
        struct[0].Gy[29,29] = -1
        struct[0].Gy[29,48] = -0.0636395371726713
        struct[0].Gy[29,49] = -0.0382049854269434
        struct[0].Gy[29,50] = -0.0636395371726737
        struct[0].Gy[29,51] = -0.0382049854269442
        struct[0].Gy[29,52] = 0.127279074345345
        struct[0].Gy[29,53] = 0.0764099708538875
        struct[0].Gy[29,54] = -0.0636395132474596
        struct[0].Gy[29,55] = -0.0382048231043602
        struct[0].Gy[29,56] = -0.0636395132474621
        struct[0].Gy[29,57] = -0.0382048231043610
        struct[0].Gy[29,58] = 0.127279026494922
        struct[0].Gy[29,59] = 0.0764096462087212
        struct[0].Gy[29,60] = -0.0636394414713370
        struct[0].Gy[29,61] = -0.0382043361371476
        struct[0].Gy[29,62] = -0.0636394414713394
        struct[0].Gy[29,63] = -0.0382043361371484
        struct[0].Gy[29,64] = 0.127278882942676
        struct[0].Gy[29,65] = 0.0764086722742961
        struct[0].Gy[29,68] = -3.19497213887036
        struct[0].Gy[29,69] = -1.91798392779206
        struct[0].Gy[29,70] = 3.19497213887045
        struct[0].Gy[29,71] = 1.91798392779192
        struct[0].Gy[30,30] = -1
        struct[0].Gy[30,48] = 0.0318681229122168
        struct[0].Gy[30,49] = -0.100328108879386
        struct[0].Gy[30,50] = -0.0318681229122152
        struct[0].Gy[30,51] = 0.100328108879391
        struct[0].Gy[30,54] = 0.0305931173773991
        struct[0].Gy[30,55] = -0.0992822970142200
        struct[0].Gy[30,56] = -0.0305931173773975
        struct[0].Gy[30,57] = 0.0992822970142252
        struct[0].Gy[30,60] = 0.0293176867112793
        struct[0].Gy[30,61] = -0.0982363113431474
        struct[0].Gy[30,62] = -0.0293176867112777
        struct[0].Gy[30,63] = 0.0982363113431526
        struct[0].Gy[30,66] = 11.6035242966407
        struct[0].Gy[30,67] = -3.08738963687029
        struct[0].Gy[30,68] = 10.1957014483333
        struct[0].Gy[30,69] = 1.79198075607073
        struct[0].Gy[30,70] = 10.1957014483333
        struct[0].Gy[30,71] = 1.79198075607047
        struct[0].Gy[31,31] = -1
        struct[0].Gy[31,48] = 0.100328108879386
        struct[0].Gy[31,49] = 0.0318681229122168
        struct[0].Gy[31,50] = -0.100328108879391
        struct[0].Gy[31,51] = -0.0318681229122152
        struct[0].Gy[31,54] = 0.0992822970142200
        struct[0].Gy[31,55] = 0.0305931173773991
        struct[0].Gy[31,56] = -0.0992822970142252
        struct[0].Gy[31,57] = -0.0305931173773975
        struct[0].Gy[31,60] = 0.0982363113431474
        struct[0].Gy[31,61] = 0.0293176867112793
        struct[0].Gy[31,62] = -0.0982363113431526
        struct[0].Gy[31,63] = -0.0293176867112777
        struct[0].Gy[31,66] = 3.08738963687029
        struct[0].Gy[31,67] = 11.6035242966407
        struct[0].Gy[31,68] = -1.79198075607073
        struct[0].Gy[31,69] = 10.1957014483333
        struct[0].Gy[31,70] = -1.79198075607047
        struct[0].Gy[31,71] = 10.1957014483333
        struct[0].Gy[32,32] = -1
        struct[0].Gy[32,50] = 0.0318681229122188
        struct[0].Gy[32,51] = -0.100328108879391
        struct[0].Gy[32,52] = -0.0318681229122203
        struct[0].Gy[32,53] = 0.100328108879387
        struct[0].Gy[32,56] = 0.0305931173774011
        struct[0].Gy[32,57] = -0.0992822970142250
        struct[0].Gy[32,58] = -0.0305931173774025
        struct[0].Gy[32,59] = 0.0992822970142214
        struct[0].Gy[32,62] = 0.0293176867112812
        struct[0].Gy[32,63] = -0.0982363113431523
        struct[0].Gy[32,64] = -0.0293176867112828
        struct[0].Gy[32,65] = 0.0982363113431489
        struct[0].Gy[32,66] = 10.1957014483335
        struct[0].Gy[32,67] = 1.79198075607090
        struct[0].Gy[32,68] = 11.6035242966410
        struct[0].Gy[32,69] = -3.08738963687010
        struct[0].Gy[32,70] = 10.1957014483334
        struct[0].Gy[32,71] = 1.79198075607072
        struct[0].Gy[33,33] = -1
        struct[0].Gy[33,50] = 0.100328108879391
        struct[0].Gy[33,51] = 0.0318681229122188
        struct[0].Gy[33,52] = -0.100328108879387
        struct[0].Gy[33,53] = -0.0318681229122203
        struct[0].Gy[33,56] = 0.0992822970142250
        struct[0].Gy[33,57] = 0.0305931173774011
        struct[0].Gy[33,58] = -0.0992822970142214
        struct[0].Gy[33,59] = -0.0305931173774025
        struct[0].Gy[33,62] = 0.0982363113431523
        struct[0].Gy[33,63] = 0.0293176867112812
        struct[0].Gy[33,64] = -0.0982363113431489
        struct[0].Gy[33,65] = -0.0293176867112828
        struct[0].Gy[33,66] = -1.79198075607090
        struct[0].Gy[33,67] = 10.1957014483335
        struct[0].Gy[33,68] = 3.08738963687010
        struct[0].Gy[33,69] = 11.6035242966410
        struct[0].Gy[33,70] = -1.79198075607072
        struct[0].Gy[33,71] = 10.1957014483334
        struct[0].Gy[34,34] = -1
        struct[0].Gy[34,48] = -0.0318681229122163
        struct[0].Gy[34,49] = 0.100328108879394
        struct[0].Gy[34,52] = 0.0318681229122155
        struct[0].Gy[34,53] = -0.100328108879396
        struct[0].Gy[34,54] = -0.0305931173773985
        struct[0].Gy[34,55] = 0.0992822970142275
        struct[0].Gy[34,58] = 0.0305931173773979
        struct[0].Gy[34,59] = -0.0992822970142302
        struct[0].Gy[34,60] = -0.0293176867112786
        struct[0].Gy[34,61] = 0.0982363113431550
        struct[0].Gy[34,64] = 0.0293176867112779
        struct[0].Gy[34,65] = -0.0982363113431575
        struct[0].Gy[34,66] = 10.1957014483334
        struct[0].Gy[34,67] = 1.79198075607079
        struct[0].Gy[34,68] = 10.1957014483334
        struct[0].Gy[34,69] = 1.79198075607092
        struct[0].Gy[34,70] = 11.6035242966408
        struct[0].Gy[34,71] = -3.08738963687035
        struct[0].Gy[35,35] = -1
        struct[0].Gy[35,48] = -0.100328108879394
        struct[0].Gy[35,49] = -0.0318681229122163
        struct[0].Gy[35,52] = 0.100328108879396
        struct[0].Gy[35,53] = 0.0318681229122155
        struct[0].Gy[35,54] = -0.0992822970142275
        struct[0].Gy[35,55] = -0.0305931173773985
        struct[0].Gy[35,58] = 0.0992822970142302
        struct[0].Gy[35,59] = 0.0305931173773979
        struct[0].Gy[35,60] = -0.0982363113431550
        struct[0].Gy[35,61] = -0.0293176867112786
        struct[0].Gy[35,64] = 0.0982363113431575
        struct[0].Gy[35,65] = 0.0293176867112779
        struct[0].Gy[35,66] = -1.79198075607079
        struct[0].Gy[35,67] = 10.1957014483334
        struct[0].Gy[35,68] = -1.79198075607092
        struct[0].Gy[35,69] = 10.1957014483334
        struct[0].Gy[35,70] = 3.08738963687035
        struct[0].Gy[35,71] = 11.6035242966408
        struct[0].Gy[36,36] = -1
        struct[0].Gy[36,48] = 0.0305931173773991
        struct[0].Gy[36,49] = -0.0992822970142201
        struct[0].Gy[36,50] = -0.0305931173773975
        struct[0].Gy[36,51] = 0.0992822970142253
        struct[0].Gy[36,54] = 0.0305929048117480
        struct[0].Gy[36,55] = -0.0992822101112667
        struct[0].Gy[36,56] = -0.0305929048117464
        struct[0].Gy[36,57] = 0.0992822101112719
        struct[0].Gy[36,60] = 0.0293174777213171
        struct[0].Gy[36,61] = -0.0982362237268541
        struct[0].Gy[36,62] = -0.0293174777213155
        struct[0].Gy[36,63] = 0.0982362237268593
        struct[0].Gy[36,66] = 11.6033854030100
        struct[0].Gy[36,67] = -3.08755280951041
        struct[0].Gy[36,68] = 10.1955728673526
        struct[0].Gy[36,69] = 1.79181314887337
        struct[0].Gy[36,70] = 10.1955728673526
        struct[0].Gy[36,71] = 1.79181314887312
        struct[0].Gy[37,37] = -1
        struct[0].Gy[37,48] = 0.0992822970142201
        struct[0].Gy[37,49] = 0.0305931173773991
        struct[0].Gy[37,50] = -0.0992822970142253
        struct[0].Gy[37,51] = -0.0305931173773975
        struct[0].Gy[37,54] = 0.0992822101112667
        struct[0].Gy[37,55] = 0.0305929048117480
        struct[0].Gy[37,56] = -0.0992822101112719
        struct[0].Gy[37,57] = -0.0305929048117464
        struct[0].Gy[37,60] = 0.0982362237268541
        struct[0].Gy[37,61] = 0.0293174777213171
        struct[0].Gy[37,62] = -0.0982362237268593
        struct[0].Gy[37,63] = -0.0293174777213155
        struct[0].Gy[37,66] = 3.08755280951041
        struct[0].Gy[37,67] = 11.6033854030100
        struct[0].Gy[37,68] = -1.79181314887337
        struct[0].Gy[37,69] = 10.1955728673526
        struct[0].Gy[37,70] = -1.79181314887312
        struct[0].Gy[37,71] = 10.1955728673526
        struct[0].Gy[38,38] = -1
        struct[0].Gy[38,50] = 0.0305931173774011
        struct[0].Gy[38,51] = -0.0992822970142250
        struct[0].Gy[38,52] = -0.0305931173774025
        struct[0].Gy[38,53] = 0.0992822970142215
        struct[0].Gy[38,56] = 0.0305929048117499
        struct[0].Gy[38,57] = -0.0992822101112716
        struct[0].Gy[38,58] = -0.0305929048117514
        struct[0].Gy[38,59] = 0.0992822101112681
        struct[0].Gy[38,62] = 0.0293174777213190
        struct[0].Gy[38,63] = -0.0982362237268590
        struct[0].Gy[38,64] = -0.0293174777213205
        struct[0].Gy[38,65] = 0.0982362237268555
        struct[0].Gy[38,66] = 10.1955728673528
        struct[0].Gy[38,67] = 1.79181314887354
        struct[0].Gy[38,68] = 11.6033854030104
        struct[0].Gy[38,69] = -3.08755280951023
        struct[0].Gy[38,70] = 10.1955728673527
        struct[0].Gy[38,71] = 1.79181314887337
        struct[0].Gy[39,39] = -1
        struct[0].Gy[39,50] = 0.0992822970142250
        struct[0].Gy[39,51] = 0.0305931173774011
        struct[0].Gy[39,52] = -0.0992822970142215
        struct[0].Gy[39,53] = -0.0305931173774025
        struct[0].Gy[39,56] = 0.0992822101112716
        struct[0].Gy[39,57] = 0.0305929048117499
        struct[0].Gy[39,58] = -0.0992822101112681
        struct[0].Gy[39,59] = -0.0305929048117514
        struct[0].Gy[39,62] = 0.0982362237268590
        struct[0].Gy[39,63] = 0.0293174777213190
        struct[0].Gy[39,64] = -0.0982362237268555
        struct[0].Gy[39,65] = -0.0293174777213205
        struct[0].Gy[39,66] = -1.79181314887354
        struct[0].Gy[39,67] = 10.1955728673528
        struct[0].Gy[39,68] = 3.08755280951023
        struct[0].Gy[39,69] = 11.6033854030104
        struct[0].Gy[39,70] = -1.79181314887337
        struct[0].Gy[39,71] = 10.1955728673527
        struct[0].Gy[40,40] = -1
        struct[0].Gy[40,48] = -0.0305931173773985
        struct[0].Gy[40,49] = 0.0992822970142276
        struct[0].Gy[40,52] = 0.0305931173773978
        struct[0].Gy[40,53] = -0.0992822970142302
        struct[0].Gy[40,54] = -0.0305929048117474
        struct[0].Gy[40,55] = 0.0992822101112742
        struct[0].Gy[40,58] = 0.0305929048117467
        struct[0].Gy[40,59] = -0.0992822101112768
        struct[0].Gy[40,60] = -0.0293174777213164
        struct[0].Gy[40,61] = 0.0982362237268616
        struct[0].Gy[40,64] = 0.0293174777213157
        struct[0].Gy[40,65] = -0.0982362237268642
        struct[0].Gy[40,66] = 10.1955728673527
        struct[0].Gy[40,67] = 1.79181314887342
        struct[0].Gy[40,68] = 10.1955728673527
        struct[0].Gy[40,69] = 1.79181314887355
        struct[0].Gy[40,70] = 11.6033854030101
        struct[0].Gy[40,71] = -3.08755280951048
        struct[0].Gy[41,41] = -1
        struct[0].Gy[41,48] = -0.0992822970142276
        struct[0].Gy[41,49] = -0.0305931173773985
        struct[0].Gy[41,52] = 0.0992822970142302
        struct[0].Gy[41,53] = 0.0305931173773978
        struct[0].Gy[41,54] = -0.0992822101112742
        struct[0].Gy[41,55] = -0.0305929048117474
        struct[0].Gy[41,58] = 0.0992822101112768
        struct[0].Gy[41,59] = 0.0305929048117467
        struct[0].Gy[41,60] = -0.0982362237268616
        struct[0].Gy[41,61] = -0.0293174777213164
        struct[0].Gy[41,64] = 0.0982362237268642
        struct[0].Gy[41,65] = 0.0293174777213157
        struct[0].Gy[41,66] = -1.79181314887342
        struct[0].Gy[41,67] = 10.1955728673527
        struct[0].Gy[41,68] = -1.79181314887355
        struct[0].Gy[41,69] = 10.1955728673527
        struct[0].Gy[41,70] = 3.08755280951048
        struct[0].Gy[41,71] = 11.6033854030101
        struct[0].Gy[42,42] = -1
        struct[0].Gy[42,48] = 0.0293176867112792
        struct[0].Gy[42,49] = -0.0982363113431474
        struct[0].Gy[42,50] = -0.0293176867112776
        struct[0].Gy[42,51] = 0.0982363113431526
        struct[0].Gy[42,54] = 0.0293174777213170
        struct[0].Gy[42,55] = -0.0982362237268541
        struct[0].Gy[42,56] = -0.0293174777213155
        struct[0].Gy[42,57] = 0.0982362237268593
        struct[0].Gy[42,60] = 0.0293168507523162
        struct[0].Gy[42,61] = -0.0982359608775054
        struct[0].Gy[42,62] = -0.0293168507523146
        struct[0].Gy[42,63] = 0.0982359608775105
        struct[0].Gy[42,66] = 11.6029687203683
        struct[0].Gy[42,67] = -3.08804231916211
        struct[0].Gy[42,68] = 10.1951871226168
        struct[0].Gy[42,69] = 1.79131033552716
        struct[0].Gy[42,70] = 10.1951871226167
        struct[0].Gy[42,71] = 1.79131033552690
        struct[0].Gy[43,43] = -1
        struct[0].Gy[43,48] = 0.0982363113431474
        struct[0].Gy[43,49] = 0.0293176867112792
        struct[0].Gy[43,50] = -0.0982363113431526
        struct[0].Gy[43,51] = -0.0293176867112776
        struct[0].Gy[43,54] = 0.0982362237268541
        struct[0].Gy[43,55] = 0.0293174777213170
        struct[0].Gy[43,56] = -0.0982362237268593
        struct[0].Gy[43,57] = -0.0293174777213155
        struct[0].Gy[43,60] = 0.0982359608775054
        struct[0].Gy[43,61] = 0.0293168507523162
        struct[0].Gy[43,62] = -0.0982359608775105
        struct[0].Gy[43,63] = -0.0293168507523146
        struct[0].Gy[43,66] = 3.08804231916211
        struct[0].Gy[43,67] = 11.6029687203683
        struct[0].Gy[43,68] = -1.79131033552716
        struct[0].Gy[43,69] = 10.1951871226168
        struct[0].Gy[43,70] = -1.79131033552690
        struct[0].Gy[43,71] = 10.1951871226167
        struct[0].Gy[44,44] = -1
        struct[0].Gy[44,50] = 0.0293176867112812
        struct[0].Gy[44,51] = -0.0982363113431524
        struct[0].Gy[44,52] = -0.0293176867112826
        struct[0].Gy[44,53] = 0.0982363113431489
        struct[0].Gy[44,56] = 0.0293174777213190
        struct[0].Gy[44,57] = -0.0982362237268590
        struct[0].Gy[44,58] = -0.0293174777213204
        struct[0].Gy[44,59] = 0.0982362237268555
        struct[0].Gy[44,62] = 0.0293168507523179
        struct[0].Gy[44,63] = -0.0982359608775103
        struct[0].Gy[44,64] = -0.0293168507523194
        struct[0].Gy[44,65] = 0.0982359608775067
        struct[0].Gy[44,66] = 10.1951871226169
        struct[0].Gy[44,67] = 1.79131033552732
        struct[0].Gy[44,68] = 11.6029687203686
        struct[0].Gy[44,69] = -3.08804231916193
        struct[0].Gy[44,70] = 10.1951871226169
        struct[0].Gy[44,71] = 1.79131033552715
        struct[0].Gy[45,45] = -1
        struct[0].Gy[45,50] = 0.0982363113431524
        struct[0].Gy[45,51] = 0.0293176867112812
        struct[0].Gy[45,52] = -0.0982363113431489
        struct[0].Gy[45,53] = -0.0293176867112826
        struct[0].Gy[45,56] = 0.0982362237268590
        struct[0].Gy[45,57] = 0.0293174777213190
        struct[0].Gy[45,58] = -0.0982362237268555
        struct[0].Gy[45,59] = -0.0293174777213204
        struct[0].Gy[45,62] = 0.0982359608775103
        struct[0].Gy[45,63] = 0.0293168507523179
        struct[0].Gy[45,64] = -0.0982359608775067
        struct[0].Gy[45,65] = -0.0293168507523194
        struct[0].Gy[45,66] = -1.79131033552732
        struct[0].Gy[45,67] = 10.1951871226169
        struct[0].Gy[45,68] = 3.08804231916193
        struct[0].Gy[45,69] = 11.6029687203686
        struct[0].Gy[45,70] = -1.79131033552715
        struct[0].Gy[45,71] = 10.1951871226169
        struct[0].Gy[46,46] = -1
        struct[0].Gy[46,48] = -0.0293176867112787
        struct[0].Gy[46,49] = 0.0982363113431550
        struct[0].Gy[46,52] = 0.0293176867112779
        struct[0].Gy[46,53] = -0.0982363113431576
        struct[0].Gy[46,54] = -0.0293174777213165
        struct[0].Gy[46,55] = 0.0982362237268616
        struct[0].Gy[46,58] = 0.0293174777213157
        struct[0].Gy[46,59] = -0.0982362237268642
        struct[0].Gy[46,60] = -0.0293168507523155
        struct[0].Gy[46,61] = 0.0982359608775129
        struct[0].Gy[46,64] = 0.0293168507523147
        struct[0].Gy[46,65] = -0.0982359608775155
        struct[0].Gy[46,66] = 10.1951871226168
        struct[0].Gy[46,67] = 1.79131033552721
        struct[0].Gy[46,68] = 10.1951871226169
        struct[0].Gy[46,69] = 1.79131033552733
        struct[0].Gy[46,70] = 11.6029687203684
        struct[0].Gy[46,71] = -3.08804231916218
        struct[0].Gy[47,47] = -1
        struct[0].Gy[47,48] = -0.0982363113431550
        struct[0].Gy[47,49] = -0.0293176867112787
        struct[0].Gy[47,52] = 0.0982363113431576
        struct[0].Gy[47,53] = 0.0293176867112779
        struct[0].Gy[47,54] = -0.0982362237268616
        struct[0].Gy[47,55] = -0.0293174777213165
        struct[0].Gy[47,58] = 0.0982362237268642
        struct[0].Gy[47,59] = 0.0293174777213157
        struct[0].Gy[47,60] = -0.0982359608775129
        struct[0].Gy[47,61] = -0.0293168507523155
        struct[0].Gy[47,64] = 0.0982359608775155
        struct[0].Gy[47,65] = 0.0293168507523147
        struct[0].Gy[47,66] = -1.79131033552721
        struct[0].Gy[47,67] = 10.1951871226168
        struct[0].Gy[47,68] = -1.79131033552733
        struct[0].Gy[47,69] = 10.1951871226169
        struct[0].Gy[47,70] = 3.08804231916218
        struct[0].Gy[47,71] = 11.6029687203684
        struct[0].Gy[48,0] = i_W1lv_a_r
        struct[0].Gy[48,1] = i_W1lv_a_i
        struct[0].Gy[48,48] = v_W1lv_a_r
        struct[0].Gy[48,49] = v_W1lv_a_i
        struct[0].Gy[49,2] = i_W1lv_b_r
        struct[0].Gy[49,3] = i_W1lv_b_i
        struct[0].Gy[49,50] = v_W1lv_b_r
        struct[0].Gy[49,51] = v_W1lv_b_i
        struct[0].Gy[50,4] = i_W1lv_c_r
        struct[0].Gy[50,5] = i_W1lv_c_i
        struct[0].Gy[50,52] = v_W1lv_c_r
        struct[0].Gy[50,53] = v_W1lv_c_i
        struct[0].Gy[51,0] = -i_W1lv_a_i
        struct[0].Gy[51,1] = i_W1lv_a_r
        struct[0].Gy[51,48] = v_W1lv_a_i
        struct[0].Gy[51,49] = -v_W1lv_a_r
        struct[0].Gy[52,2] = -i_W1lv_b_i
        struct[0].Gy[52,3] = i_W1lv_b_r
        struct[0].Gy[52,50] = v_W1lv_b_i
        struct[0].Gy[52,51] = -v_W1lv_b_r
        struct[0].Gy[53,4] = -i_W1lv_c_i
        struct[0].Gy[53,5] = i_W1lv_c_r
        struct[0].Gy[53,52] = v_W1lv_c_i
        struct[0].Gy[53,53] = -v_W1lv_c_r
        struct[0].Gy[54,6] = i_W2lv_a_r
        struct[0].Gy[54,7] = i_W2lv_a_i
        struct[0].Gy[54,54] = v_W2lv_a_r
        struct[0].Gy[54,55] = v_W2lv_a_i
        struct[0].Gy[55,8] = i_W2lv_b_r
        struct[0].Gy[55,9] = i_W2lv_b_i
        struct[0].Gy[55,56] = v_W2lv_b_r
        struct[0].Gy[55,57] = v_W2lv_b_i
        struct[0].Gy[56,10] = i_W2lv_c_r
        struct[0].Gy[56,11] = i_W2lv_c_i
        struct[0].Gy[56,58] = v_W2lv_c_r
        struct[0].Gy[56,59] = v_W2lv_c_i
        struct[0].Gy[57,6] = -i_W2lv_a_i
        struct[0].Gy[57,7] = i_W2lv_a_r
        struct[0].Gy[57,54] = v_W2lv_a_i
        struct[0].Gy[57,55] = -v_W2lv_a_r
        struct[0].Gy[58,8] = -i_W2lv_b_i
        struct[0].Gy[58,9] = i_W2lv_b_r
        struct[0].Gy[58,56] = v_W2lv_b_i
        struct[0].Gy[58,57] = -v_W2lv_b_r
        struct[0].Gy[59,10] = -i_W2lv_c_i
        struct[0].Gy[59,11] = i_W2lv_c_r
        struct[0].Gy[59,58] = v_W2lv_c_i
        struct[0].Gy[59,59] = -v_W2lv_c_r
        struct[0].Gy[60,12] = i_W3lv_a_r
        struct[0].Gy[60,13] = i_W3lv_a_i
        struct[0].Gy[60,60] = v_W3lv_a_r
        struct[0].Gy[60,61] = v_W3lv_a_i
        struct[0].Gy[61,14] = i_W3lv_b_r
        struct[0].Gy[61,15] = i_W3lv_b_i
        struct[0].Gy[61,62] = v_W3lv_b_r
        struct[0].Gy[61,63] = v_W3lv_b_i
        struct[0].Gy[62,16] = i_W3lv_c_r
        struct[0].Gy[62,17] = i_W3lv_c_i
        struct[0].Gy[62,64] = v_W3lv_c_r
        struct[0].Gy[62,65] = v_W3lv_c_i
        struct[0].Gy[63,12] = -i_W3lv_a_i
        struct[0].Gy[63,13] = i_W3lv_a_r
        struct[0].Gy[63,60] = v_W3lv_a_i
        struct[0].Gy[63,61] = -v_W3lv_a_r
        struct[0].Gy[64,14] = -i_W3lv_b_i
        struct[0].Gy[64,15] = i_W3lv_b_r
        struct[0].Gy[64,62] = v_W3lv_b_i
        struct[0].Gy[64,63] = -v_W3lv_b_r
        struct[0].Gy[65,16] = -i_W3lv_c_i
        struct[0].Gy[65,17] = i_W3lv_c_r
        struct[0].Gy[65,64] = v_W3lv_c_i
        struct[0].Gy[65,65] = -v_W3lv_c_r
        struct[0].Gy[66,18] = i_POImv_a_r
        struct[0].Gy[66,19] = i_POImv_a_i
        struct[0].Gy[66,66] = v_POImv_a_r
        struct[0].Gy[66,67] = v_POImv_a_i
        struct[0].Gy[67,20] = i_POImv_b_r
        struct[0].Gy[67,21] = i_POImv_b_i
        struct[0].Gy[67,68] = v_POImv_b_r
        struct[0].Gy[67,69] = v_POImv_b_i
        struct[0].Gy[68,22] = i_POImv_c_r
        struct[0].Gy[68,23] = i_POImv_c_i
        struct[0].Gy[68,70] = v_POImv_c_r
        struct[0].Gy[68,71] = v_POImv_c_i
        struct[0].Gy[69,18] = -i_POImv_a_i
        struct[0].Gy[69,19] = i_POImv_a_r
        struct[0].Gy[69,66] = v_POImv_a_i
        struct[0].Gy[69,67] = -v_POImv_a_r
        struct[0].Gy[70,20] = -i_POImv_b_i
        struct[0].Gy[70,21] = i_POImv_b_r
        struct[0].Gy[70,68] = v_POImv_b_i
        struct[0].Gy[70,69] = -v_POImv_b_r
        struct[0].Gy[71,22] = -i_POImv_c_i
        struct[0].Gy[71,23] = i_POImv_c_r
        struct[0].Gy[71,70] = v_POImv_c_i
        struct[0].Gy[71,71] = -v_POImv_c_r

    if mode > 12:


        struct[0].Gu[0,0] = 0.00697522388628822
        struct[0].Gu[0,1] = 0.00000171338624816169
        struct[0].Gu[0,2] = -0.00348761194314407
        struct[0].Gu[0,3] = -8.56693124078082E-7
        struct[0].Gu[0,4] = -0.00348761194314416
        struct[0].Gu[0,5] = -8.56693124085475E-7
        struct[0].Gu[0,6] = 0.0764099708538850
        struct[0].Gu[0,7] = -0.127279074345343
        struct[0].Gu[0,8] = -0.0382049854269420
        struct[0].Gu[0,9] = 0.0636395371726705
        struct[0].Gu[0,10] = -0.0382049854269431
        struct[0].Gu[0,11] = 0.0636395371726721
        struct[0].Gu[0,12] = 0.0318681229122137
        struct[0].Gu[0,13] = -0.100328108879392
        struct[0].Gu[0,16] = -0.0318681229122191
        struct[0].Gu[0,17] = 0.100328108879387
        struct[0].Gu[0,18] = 0.0305931173773959
        struct[0].Gu[0,19] = -0.0992822970142266
        struct[0].Gu[0,22] = -0.0305931173774017
        struct[0].Gu[0,23] = 0.0992822970142214
        struct[0].Gu[0,24] = 0.0293176867112763
        struct[0].Gu[0,25] = -0.0982363113431537
        struct[0].Gu[0,28] = -0.0293176867112818
        struct[0].Gu[0,29] = 0.0982363113431486
        struct[0].Gu[1,0] = -0.00000171338624816169
        struct[0].Gu[1,1] = 0.00697522388628822
        struct[0].Gu[1,2] = 8.56693124078082E-7
        struct[0].Gu[1,3] = -0.00348761194314407
        struct[0].Gu[1,4] = 8.56693124085475E-7
        struct[0].Gu[1,5] = -0.00348761194314416
        struct[0].Gu[1,6] = 0.127279074345343
        struct[0].Gu[1,7] = 0.0764099708538850
        struct[0].Gu[1,8] = -0.0636395371726705
        struct[0].Gu[1,9] = -0.0382049854269420
        struct[0].Gu[1,10] = -0.0636395371726721
        struct[0].Gu[1,11] = -0.0382049854269431
        struct[0].Gu[1,12] = 0.100328108879392
        struct[0].Gu[1,13] = 0.0318681229122137
        struct[0].Gu[1,16] = -0.100328108879387
        struct[0].Gu[1,17] = -0.0318681229122191
        struct[0].Gu[1,18] = 0.0992822970142266
        struct[0].Gu[1,19] = 0.0305931173773959
        struct[0].Gu[1,22] = -0.0992822970142214
        struct[0].Gu[1,23] = -0.0305931173774017
        struct[0].Gu[1,24] = 0.0982363113431537
        struct[0].Gu[1,25] = 0.0293176867112763
        struct[0].Gu[1,28] = -0.0982363113431486
        struct[0].Gu[1,29] = -0.0293176867112818
        struct[0].Gu[2,0] = -0.00348761194314409
        struct[0].Gu[2,1] = -8.56693124042750E-7
        struct[0].Gu[2,2] = 0.00697522388628832
        struct[0].Gu[2,3] = 0.00000171338624816775
        struct[0].Gu[2,4] = -0.00348761194314422
        struct[0].Gu[2,5] = -8.56693124121928E-7
        struct[0].Gu[2,6] = -0.0382049854269416
        struct[0].Gu[2,7] = 0.0636395371726714
        struct[0].Gu[2,8] = 0.0764099708538862
        struct[0].Gu[2,9] = -0.127279074345344
        struct[0].Gu[2,10] = -0.0382049854269445
        struct[0].Gu[2,11] = 0.0636395371726728
        struct[0].Gu[2,12] = -0.0318681229122115
        struct[0].Gu[2,13] = 0.100328108879395
        struct[0].Gu[2,14] = 0.0318681229122223
        struct[0].Gu[2,15] = -0.100328108879388
        struct[0].Gu[2,18] = -0.0305931173773937
        struct[0].Gu[2,19] = 0.0992822970142287
        struct[0].Gu[2,20] = 0.0305931173774048
        struct[0].Gu[2,21] = -0.0992822970142215
        struct[0].Gu[2,24] = -0.0293176867112742
        struct[0].Gu[2,25] = 0.0982363113431560
        struct[0].Gu[2,26] = 0.0293176867112850
        struct[0].Gu[2,27] = -0.0982363113431489
        struct[0].Gu[3,0] = 8.56693124042750E-7
        struct[0].Gu[3,1] = -0.00348761194314409
        struct[0].Gu[3,2] = -0.00000171338624816775
        struct[0].Gu[3,3] = 0.00697522388628832
        struct[0].Gu[3,4] = 8.56693124121928E-7
        struct[0].Gu[3,5] = -0.00348761194314422
        struct[0].Gu[3,6] = -0.0636395371726714
        struct[0].Gu[3,7] = -0.0382049854269416
        struct[0].Gu[3,8] = 0.127279074345344
        struct[0].Gu[3,9] = 0.0764099708538862
        struct[0].Gu[3,10] = -0.0636395371726728
        struct[0].Gu[3,11] = -0.0382049854269445
        struct[0].Gu[3,12] = -0.100328108879395
        struct[0].Gu[3,13] = -0.0318681229122115
        struct[0].Gu[3,14] = 0.100328108879388
        struct[0].Gu[3,15] = 0.0318681229122223
        struct[0].Gu[3,18] = -0.0992822970142287
        struct[0].Gu[3,19] = -0.0305931173773937
        struct[0].Gu[3,20] = 0.0992822970142215
        struct[0].Gu[3,21] = 0.0305931173774048
        struct[0].Gu[3,24] = -0.0982363113431560
        struct[0].Gu[3,25] = -0.0293176867112742
        struct[0].Gu[3,26] = 0.0982363113431489
        struct[0].Gu[3,27] = 0.0293176867112850
        struct[0].Gu[4,0] = -0.00348761194314413
        struct[0].Gu[4,1] = -8.56693124118069E-7
        struct[0].Gu[4,2] = -0.00348761194314425
        struct[0].Gu[4,3] = -8.56693124090318E-7
        struct[0].Gu[4,4] = 0.00697522388628838
        struct[0].Gu[4,5] = 0.00000171338624820762
        struct[0].Gu[4,6] = -0.0382049854269434
        struct[0].Gu[4,7] = 0.0636395371726713
        struct[0].Gu[4,8] = -0.0382049854269442
        struct[0].Gu[4,9] = 0.0636395371726737
        struct[0].Gu[4,10] = 0.0764099708538876
        struct[0].Gu[4,11] = -0.127279074345345
        struct[0].Gu[4,14] = -0.0318681229122202
        struct[0].Gu[4,15] = 0.100328108879391
        struct[0].Gu[4,16] = 0.0318681229122156
        struct[0].Gu[4,17] = -0.100328108879392
        struct[0].Gu[4,20] = -0.0305931173774027
        struct[0].Gu[4,21] = 0.0992822970142254
        struct[0].Gu[4,22] = 0.0305931173773979
        struct[0].Gu[4,23] = -0.0992822970142266
        struct[0].Gu[4,26] = -0.0293176867112827
        struct[0].Gu[4,27] = 0.0982363113431527
        struct[0].Gu[4,28] = 0.0293176867112781
        struct[0].Gu[4,29] = -0.0982363113431540
        struct[0].Gu[5,0] = 8.56693124118069E-7
        struct[0].Gu[5,1] = -0.00348761194314413
        struct[0].Gu[5,2] = 8.56693124090318E-7
        struct[0].Gu[5,3] = -0.00348761194314425
        struct[0].Gu[5,4] = -0.00000171338624820762
        struct[0].Gu[5,5] = 0.00697522388628838
        struct[0].Gu[5,6] = -0.0636395371726713
        struct[0].Gu[5,7] = -0.0382049854269434
        struct[0].Gu[5,8] = -0.0636395371726737
        struct[0].Gu[5,9] = -0.0382049854269442
        struct[0].Gu[5,10] = 0.127279074345345
        struct[0].Gu[5,11] = 0.0764099708538876
        struct[0].Gu[5,14] = -0.100328108879391
        struct[0].Gu[5,15] = -0.0318681229122202
        struct[0].Gu[5,16] = 0.100328108879392
        struct[0].Gu[5,17] = 0.0318681229122156
        struct[0].Gu[5,20] = -0.0992822970142254
        struct[0].Gu[5,21] = -0.0305931173774027
        struct[0].Gu[5,22] = 0.0992822970142266
        struct[0].Gu[5,23] = 0.0305931173773979
        struct[0].Gu[5,26] = -0.0982363113431527
        struct[0].Gu[5,27] = -0.0293176867112827
        struct[0].Gu[5,28] = 0.0982363113431540
        struct[0].Gu[5,29] = 0.0293176867112781
        struct[0].Gu[6,0] = 0.00697521411040091
        struct[0].Gu[6,1] = 0.00000170146300431068
        struct[0].Gu[6,2] = -0.00348760705520041
        struct[0].Gu[6,3] = -8.50731502151710E-7
        struct[0].Gu[6,4] = -0.00348760705520050
        struct[0].Gu[6,5] = -8.50731502159103E-7
        struct[0].Gu[6,6] = 0.0764096462087187
        struct[0].Gu[6,7] = -0.127279026494919
        struct[0].Gu[6,8] = -0.0382048231043588
        struct[0].Gu[6,9] = 0.0636395132474589
        struct[0].Gu[6,10] = -0.0382048231043599
        struct[0].Gu[6,11] = 0.0636395132474605
        struct[0].Gu[6,12] = 0.0305931173773962
        struct[0].Gu[6,13] = -0.0992822970142262
        struct[0].Gu[6,16] = -0.0305931173774014
        struct[0].Gu[6,17] = 0.0992822970142214
        struct[0].Gu[6,18] = 0.0305929048117450
        struct[0].Gu[6,19] = -0.0992822101112730
        struct[0].Gu[6,22] = -0.0305929048117504
        struct[0].Gu[6,23] = 0.0992822101112681
        struct[0].Gu[6,24] = 0.0293174777213142
        struct[0].Gu[6,25] = -0.0982362237268603
        struct[0].Gu[6,28] = -0.0293174777213195
        struct[0].Gu[6,29] = 0.0982362237268554
        struct[0].Gu[7,0] = -0.00000170146300431068
        struct[0].Gu[7,1] = 0.00697521411040091
        struct[0].Gu[7,2] = 8.50731502151710E-7
        struct[0].Gu[7,3] = -0.00348760705520041
        struct[0].Gu[7,4] = 8.50731502159103E-7
        struct[0].Gu[7,5] = -0.00348760705520050
        struct[0].Gu[7,6] = 0.127279026494919
        struct[0].Gu[7,7] = 0.0764096462087187
        struct[0].Gu[7,8] = -0.0636395132474589
        struct[0].Gu[7,9] = -0.0382048231043588
        struct[0].Gu[7,10] = -0.0636395132474605
        struct[0].Gu[7,11] = -0.0382048231043599
        struct[0].Gu[7,12] = 0.0992822970142262
        struct[0].Gu[7,13] = 0.0305931173773962
        struct[0].Gu[7,16] = -0.0992822970142214
        struct[0].Gu[7,17] = -0.0305931173774014
        struct[0].Gu[7,18] = 0.0992822101112730
        struct[0].Gu[7,19] = 0.0305929048117450
        struct[0].Gu[7,22] = -0.0992822101112681
        struct[0].Gu[7,23] = -0.0305929048117504
        struct[0].Gu[7,24] = 0.0982362237268603
        struct[0].Gu[7,25] = 0.0293174777213142
        struct[0].Gu[7,28] = -0.0982362237268554
        struct[0].Gu[7,29] = -0.0293174777213195
        struct[0].Gu[8,0] = -0.00348760705520044
        struct[0].Gu[8,1] = -8.50731502117679E-7
        struct[0].Gu[8,2] = 0.00697521411040101
        struct[0].Gu[8,3] = 0.00000170146300431631
        struct[0].Gu[8,4] = -0.00348760705520057
        struct[0].Gu[8,5] = -8.50731502195773E-7
        struct[0].Gu[8,6] = -0.0382048231043585
        struct[0].Gu[8,7] = 0.0636395132474598
        struct[0].Gu[8,8] = 0.0764096462087198
        struct[0].Gu[8,9] = -0.127279026494921
        struct[0].Gu[8,10] = -0.0382048231043613
        struct[0].Gu[8,11] = 0.0636395132474612
        struct[0].Gu[8,12] = -0.0305931173773939
        struct[0].Gu[8,13] = 0.0992822970142286
        struct[0].Gu[8,14] = 0.0305931173774045
        struct[0].Gu[8,15] = -0.0992822970142216
        struct[0].Gu[8,18] = -0.0305929048117428
        struct[0].Gu[8,19] = 0.0992822101112753
        struct[0].Gu[8,20] = 0.0305929048117536
        struct[0].Gu[8,21] = -0.0992822101112682
        struct[0].Gu[8,24] = -0.0293174777213120
        struct[0].Gu[8,25] = 0.0982362237268626
        struct[0].Gu[8,26] = 0.0293174777213226
        struct[0].Gu[8,27] = -0.0982362237268556
        struct[0].Gu[9,0] = 8.50731502117679E-7
        struct[0].Gu[9,1] = -0.00348760705520044
        struct[0].Gu[9,2] = -0.00000170146300431631
        struct[0].Gu[9,3] = 0.00697521411040101
        struct[0].Gu[9,4] = 8.50731502195773E-7
        struct[0].Gu[9,5] = -0.00348760705520057
        struct[0].Gu[9,6] = -0.0636395132474598
        struct[0].Gu[9,7] = -0.0382048231043585
        struct[0].Gu[9,8] = 0.127279026494921
        struct[0].Gu[9,9] = 0.0764096462087198
        struct[0].Gu[9,10] = -0.0636395132474612
        struct[0].Gu[9,11] = -0.0382048231043613
        struct[0].Gu[9,12] = -0.0992822970142286
        struct[0].Gu[9,13] = -0.0305931173773939
        struct[0].Gu[9,14] = 0.0992822970142216
        struct[0].Gu[9,15] = 0.0305931173774045
        struct[0].Gu[9,18] = -0.0992822101112753
        struct[0].Gu[9,19] = -0.0305929048117428
        struct[0].Gu[9,20] = 0.0992822101112682
        struct[0].Gu[9,21] = 0.0305929048117536
        struct[0].Gu[9,24] = -0.0982362237268626
        struct[0].Gu[9,25] = -0.0293174777213120
        struct[0].Gu[9,26] = 0.0982362237268556
        struct[0].Gu[9,27] = 0.0293174777213226
        struct[0].Gu[10,0] = -0.00348760705520048
        struct[0].Gu[10,1] = -8.50731502192782E-7
        struct[0].Gu[10,2] = -0.00348760705520059
        struct[0].Gu[10,3] = -8.50731502164163E-7
        struct[0].Gu[10,4] = 0.00697521411040107
        struct[0].Gu[10,5] = 0.00000170146300435488
        struct[0].Gu[10,6] = -0.0382048231043602
        struct[0].Gu[10,7] = 0.0636395132474597
        struct[0].Gu[10,8] = -0.0382048231043610
        struct[0].Gu[10,9] = 0.0636395132474621
        struct[0].Gu[10,10] = 0.0764096462087212
        struct[0].Gu[10,11] = -0.127279026494922
        struct[0].Gu[10,14] = -0.0305931173774024
        struct[0].Gu[10,15] = 0.0992822970142252
        struct[0].Gu[10,16] = 0.0305931173773979
        struct[0].Gu[10,17] = -0.0992822970142263
        struct[0].Gu[10,20] = -0.0305929048117514
        struct[0].Gu[10,21] = 0.0992822101112720
        struct[0].Gu[10,22] = 0.0305929048117468
        struct[0].Gu[10,23] = -0.0992822101112732
        struct[0].Gu[10,26] = -0.0293174777213204
        struct[0].Gu[10,27] = 0.0982362237268593
        struct[0].Gu[10,28] = 0.0293174777213159
        struct[0].Gu[10,29] = -0.0982362237268605
        struct[0].Gu[11,0] = 8.50731502192782E-7
        struct[0].Gu[11,1] = -0.00348760705520048
        struct[0].Gu[11,2] = 8.50731502164163E-7
        struct[0].Gu[11,3] = -0.00348760705520059
        struct[0].Gu[11,4] = -0.00000170146300435488
        struct[0].Gu[11,5] = 0.00697521411040107
        struct[0].Gu[11,6] = -0.0636395132474597
        struct[0].Gu[11,7] = -0.0382048231043602
        struct[0].Gu[11,8] = -0.0636395132474621
        struct[0].Gu[11,9] = -0.0382048231043610
        struct[0].Gu[11,10] = 0.127279026494922
        struct[0].Gu[11,11] = 0.0764096462087212
        struct[0].Gu[11,14] = -0.0992822970142252
        struct[0].Gu[11,15] = -0.0305931173774024
        struct[0].Gu[11,16] = 0.0992822970142263
        struct[0].Gu[11,17] = 0.0305931173773979
        struct[0].Gu[11,20] = -0.0992822101112720
        struct[0].Gu[11,21] = -0.0305929048117514
        struct[0].Gu[11,22] = 0.0992822101112732
        struct[0].Gu[11,23] = 0.0305929048117468
        struct[0].Gu[11,26] = -0.0982362237268593
        struct[0].Gu[11,27] = -0.0293174777213204
        struct[0].Gu[11,28] = 0.0982362237268605
        struct[0].Gu[11,29] = 0.0293174777213159
        struct[0].Gu[12,0] = 0.00697518478272564
        struct[0].Gu[12,1] = 0.00000166569333960609
        struct[0].Gu[12,2] = -0.00348759239136277
        struct[0].Gu[12,3] = -8.32846669799631E-7
        struct[0].Gu[12,4] = -0.00348759239136286
        struct[0].Gu[12,5] = -8.32846669806590E-7
        struct[0].Gu[12,6] = 0.0764086722742936
        struct[0].Gu[12,7] = -0.127278882942674
        struct[0].Gu[12,8] = -0.0382043361371463
        struct[0].Gu[12,9] = 0.0636394414713363
        struct[0].Gu[12,10] = -0.0382043361371473
        struct[0].Gu[12,11] = 0.0636394414713378
        struct[0].Gu[12,12] = 0.0293176867112763
        struct[0].Gu[12,13] = -0.0982363113431535
        struct[0].Gu[12,16] = -0.0293176867112815
        struct[0].Gu[12,17] = 0.0982363113431488
        struct[0].Gu[12,18] = 0.0293174777213141
        struct[0].Gu[12,19] = -0.0982362237268603
        struct[0].Gu[12,22] = -0.0293174777213194
        struct[0].Gu[12,23] = 0.0982362237268555
        struct[0].Gu[12,24] = 0.0293168507523132
        struct[0].Gu[12,25] = -0.0982359608775115
        struct[0].Gu[12,28] = -0.0293168507523184
        struct[0].Gu[12,29] = 0.0982359608775067
        struct[0].Gu[13,0] = -0.00000166569333960609
        struct[0].Gu[13,1] = 0.00697518478272564
        struct[0].Gu[13,2] = 8.32846669799631E-7
        struct[0].Gu[13,3] = -0.00348759239136277
        struct[0].Gu[13,4] = 8.32846669806590E-7
        struct[0].Gu[13,5] = -0.00348759239136286
        struct[0].Gu[13,6] = 0.127278882942674
        struct[0].Gu[13,7] = 0.0764086722742936
        struct[0].Gu[13,8] = -0.0636394414713363
        struct[0].Gu[13,9] = -0.0382043361371463
        struct[0].Gu[13,10] = -0.0636394414713378
        struct[0].Gu[13,11] = -0.0382043361371473
        struct[0].Gu[13,12] = 0.0982363113431535
        struct[0].Gu[13,13] = 0.0293176867112763
        struct[0].Gu[13,16] = -0.0982363113431488
        struct[0].Gu[13,17] = -0.0293176867112815
        struct[0].Gu[13,18] = 0.0982362237268603
        struct[0].Gu[13,19] = 0.0293174777213141
        struct[0].Gu[13,22] = -0.0982362237268555
        struct[0].Gu[13,23] = -0.0293174777213194
        struct[0].Gu[13,24] = 0.0982359608775115
        struct[0].Gu[13,25] = 0.0293168507523132
        struct[0].Gu[13,28] = -0.0982359608775067
        struct[0].Gu[13,29] = -0.0293168507523184
        struct[0].Gu[14,0] = -0.00348759239136280
        struct[0].Gu[14,1] = -8.32846669765817E-7
        struct[0].Gu[14,2] = 0.00697518478272573
        struct[0].Gu[14,3] = 0.00000166569333961041
        struct[0].Gu[14,4] = -0.00348759239136293
        struct[0].Gu[14,5] = -8.32846669843044E-7
        struct[0].Gu[14,6] = -0.0382043361371459
        struct[0].Gu[14,7] = 0.0636394414713372
        struct[0].Gu[14,8] = 0.0764086722742947
        struct[0].Gu[14,9] = -0.127278882942676
        struct[0].Gu[14,10] = -0.0382043361371487
        struct[0].Gu[14,11] = 0.0636394414713386
        struct[0].Gu[14,12] = -0.0293176867112742
        struct[0].Gu[14,13] = 0.0982363113431559
        struct[0].Gu[14,14] = 0.0293176867112845
        struct[0].Gu[14,15] = -0.0982363113431490
        struct[0].Gu[14,18] = -0.0293174777213120
        struct[0].Gu[14,19] = 0.0982362237268626
        struct[0].Gu[14,20] = 0.0293174777213224
        struct[0].Gu[14,21] = -0.0982362237268557
        struct[0].Gu[14,24] = -0.0293168507523111
        struct[0].Gu[14,25] = 0.0982359608775139
        struct[0].Gu[14,26] = 0.0293168507523214
        struct[0].Gu[14,27] = -0.0982359608775069
        struct[0].Gu[15,0] = 8.32846669765817E-7
        struct[0].Gu[15,1] = -0.00348759239136280
        struct[0].Gu[15,2] = -0.00000166569333961041
        struct[0].Gu[15,3] = 0.00697518478272573
        struct[0].Gu[15,4] = 8.32846669843044E-7
        struct[0].Gu[15,5] = -0.00348759239136293
        struct[0].Gu[15,6] = -0.0636394414713372
        struct[0].Gu[15,7] = -0.0382043361371459
        struct[0].Gu[15,8] = 0.127278882942676
        struct[0].Gu[15,9] = 0.0764086722742947
        struct[0].Gu[15,10] = -0.0636394414713386
        struct[0].Gu[15,11] = -0.0382043361371487
        struct[0].Gu[15,12] = -0.0982363113431559
        struct[0].Gu[15,13] = -0.0293176867112742
        struct[0].Gu[15,14] = 0.0982363113431490
        struct[0].Gu[15,15] = 0.0293176867112845
        struct[0].Gu[15,18] = -0.0982362237268626
        struct[0].Gu[15,19] = -0.0293174777213120
        struct[0].Gu[15,20] = 0.0982362237268557
        struct[0].Gu[15,21] = 0.0293174777213224
        struct[0].Gu[15,24] = -0.0982359608775139
        struct[0].Gu[15,25] = -0.0293168507523111
        struct[0].Gu[15,26] = 0.0982359608775069
        struct[0].Gu[15,27] = 0.0293168507523214
        struct[0].Gu[16,0] = -0.00348759239136284
        struct[0].Gu[16,1] = -8.32846669840052E-7
        struct[0].Gu[16,2] = -0.00348759239136295
        struct[0].Gu[16,3] = -8.32846669810349E-7
        struct[0].Gu[16,4] = 0.00697518478272579
        struct[0].Gu[16,5] = 0.00000166569333964898
        struct[0].Gu[16,6] = -0.0382043361371477
        struct[0].Gu[16,7] = 0.0636394414713370
        struct[0].Gu[16,8] = -0.0382043361371484
        struct[0].Gu[16,9] = 0.0636394414713394
        struct[0].Gu[16,10] = 0.0764086722742961
        struct[0].Gu[16,11] = -0.127278882942676
        struct[0].Gu[16,14] = -0.0293176867112824
        struct[0].Gu[16,15] = 0.0982363113431526
        struct[0].Gu[16,16] = 0.0293176867112781
        struct[0].Gu[16,17] = -0.0982363113431537
        struct[0].Gu[16,20] = -0.0293174777213203
        struct[0].Gu[16,21] = 0.0982362237268593
        struct[0].Gu[16,22] = 0.0293174777213159
        struct[0].Gu[16,23] = -0.0982362237268605
        struct[0].Gu[16,26] = -0.0293168507523193
        struct[0].Gu[16,27] = 0.0982359608775105
        struct[0].Gu[16,28] = 0.0293168507523149
        struct[0].Gu[16,29] = -0.0982359608775118
        struct[0].Gu[17,0] = 8.32846669840052E-7
        struct[0].Gu[17,1] = -0.00348759239136284
        struct[0].Gu[17,2] = 8.32846669810349E-7
        struct[0].Gu[17,3] = -0.00348759239136295
        struct[0].Gu[17,4] = -0.00000166569333964898
        struct[0].Gu[17,5] = 0.00697518478272579
        struct[0].Gu[17,6] = -0.0636394414713370
        struct[0].Gu[17,7] = -0.0382043361371477
        struct[0].Gu[17,8] = -0.0636394414713394
        struct[0].Gu[17,9] = -0.0382043361371484
        struct[0].Gu[17,10] = 0.127278882942676
        struct[0].Gu[17,11] = 0.0764086722742961
        struct[0].Gu[17,14] = -0.0982363113431526
        struct[0].Gu[17,15] = -0.0293176867112824
        struct[0].Gu[17,16] = 0.0982363113431537
        struct[0].Gu[17,17] = 0.0293176867112781
        struct[0].Gu[17,20] = -0.0982362237268593
        struct[0].Gu[17,21] = -0.0293174777213203
        struct[0].Gu[17,22] = 0.0982362237268605
        struct[0].Gu[17,23] = 0.0293174777213159
        struct[0].Gu[17,26] = -0.0982359608775105
        struct[0].Gu[17,27] = -0.0293168507523193
        struct[0].Gu[17,28] = 0.0982359608775118
        struct[0].Gu[17,29] = 0.0293168507523149
        struct[0].Gu[18,0] = 0.175091156146066
        struct[0].Gu[18,1] = 0.0000403160543824476
        struct[0].Gu[18,2] = -0.175091156146074
        struct[0].Gu[18,3] = -0.0000403160543774783
        struct[0].Gu[18,4] = 3.80491223180095E-17
        struct[0].Gu[18,5] = -3.95108865408280E-17
        struct[0].Gu[18,6] = 1.91798392779190
        struct[0].Gu[18,7] = -3.19497213887022
        struct[0].Gu[18,8] = -1.91798392779190
        struct[0].Gu[18,9] = 3.19497213887041
        struct[0].Gu[18,12] = 11.6035242966407
        struct[0].Gu[18,13] = -3.08738963687028
        struct[0].Gu[18,14] = 10.1957014483333
        struct[0].Gu[18,15] = 1.79198075607073
        struct[0].Gu[18,16] = 10.1957014483332
        struct[0].Gu[18,17] = 1.79198075607047
        struct[0].Gu[18,18] = 11.6033854030100
        struct[0].Gu[18,19] = -3.08755280951041
        struct[0].Gu[18,20] = 10.1955728673526
        struct[0].Gu[18,21] = 1.79181314887337
        struct[0].Gu[18,22] = 10.1955728673525
        struct[0].Gu[18,23] = 1.79181314887311
        struct[0].Gu[18,24] = 11.6029687203683
        struct[0].Gu[18,25] = -3.08804231916211
        struct[0].Gu[18,26] = 10.1951871226168
        struct[0].Gu[18,27] = 1.79131033552716
        struct[0].Gu[18,28] = 10.1951871226167
        struct[0].Gu[18,29] = 1.79131033552690
        struct[0].Gu[19,0] = -0.0000403160543824476
        struct[0].Gu[19,1] = 0.175091156146066
        struct[0].Gu[19,2] = 0.0000403160543774783
        struct[0].Gu[19,3] = -0.175091156146074
        struct[0].Gu[19,4] = 3.95108865408280E-17
        struct[0].Gu[19,5] = 3.80491223180095E-17
        struct[0].Gu[19,6] = 3.19497213887022
        struct[0].Gu[19,7] = 1.91798392779190
        struct[0].Gu[19,8] = -3.19497213887041
        struct[0].Gu[19,9] = -1.91798392779190
        struct[0].Gu[19,12] = 3.08738963687028
        struct[0].Gu[19,13] = 11.6035242966407
        struct[0].Gu[19,14] = -1.79198075607073
        struct[0].Gu[19,15] = 10.1957014483333
        struct[0].Gu[19,16] = -1.79198075607047
        struct[0].Gu[19,17] = 10.1957014483332
        struct[0].Gu[19,18] = 3.08755280951041
        struct[0].Gu[19,19] = 11.6033854030100
        struct[0].Gu[19,20] = -1.79181314887337
        struct[0].Gu[19,21] = 10.1955728673526
        struct[0].Gu[19,22] = -1.79181314887311
        struct[0].Gu[19,23] = 10.1955728673525
        struct[0].Gu[19,24] = 3.08804231916211
        struct[0].Gu[19,25] = 11.6029687203683
        struct[0].Gu[19,26] = -1.79131033552716
        struct[0].Gu[19,27] = 10.1951871226168
        struct[0].Gu[19,28] = -1.79131033552690
        struct[0].Gu[19,29] = 10.1951871226167
        struct[0].Gu[20,0] = 2.93587278316694E-18
        struct[0].Gu[20,1] = 4.89567608905926E-18
        struct[0].Gu[20,2] = 0.175091156146075
        struct[0].Gu[20,3] = 0.0000403160543830677
        struct[0].Gu[20,4] = -0.175091156146070
        struct[0].Gu[20,5] = -0.0000403160543870874
        struct[0].Gu[20,8] = 1.91798392779201
        struct[0].Gu[20,9] = -3.19497213887037
        struct[0].Gu[20,10] = -1.91798392779203
        struct[0].Gu[20,11] = 3.19497213887023
        struct[0].Gu[20,12] = 10.1957014483335
        struct[0].Gu[20,13] = 1.79198075607090
        struct[0].Gu[20,14] = 11.6035242966410
        struct[0].Gu[20,15] = -3.08738963687010
        struct[0].Gu[20,16] = 10.1957014483334
        struct[0].Gu[20,17] = 1.79198075607072
        struct[0].Gu[20,18] = 10.1955728673528
        struct[0].Gu[20,19] = 1.79181314887354
        struct[0].Gu[20,20] = 11.6033854030103
        struct[0].Gu[20,21] = -3.08755280951023
        struct[0].Gu[20,22] = 10.1955728673527
        struct[0].Gu[20,23] = 1.79181314887336
        struct[0].Gu[20,24] = 10.1951871226169
        struct[0].Gu[20,25] = 1.79131033552732
        struct[0].Gu[20,26] = 11.6029687203686
        struct[0].Gu[20,27] = -3.08804231916193
        struct[0].Gu[20,28] = 10.1951871226169
        struct[0].Gu[20,29] = 1.79131033552715
        struct[0].Gu[21,0] = -4.89567608905926E-18
        struct[0].Gu[21,1] = 2.93587278316694E-18
        struct[0].Gu[21,2] = -0.0000403160543830677
        struct[0].Gu[21,3] = 0.175091156146075
        struct[0].Gu[21,4] = 0.0000403160543870874
        struct[0].Gu[21,5] = -0.175091156146070
        struct[0].Gu[21,8] = 3.19497213887037
        struct[0].Gu[21,9] = 1.91798392779201
        struct[0].Gu[21,10] = -3.19497213887023
        struct[0].Gu[21,11] = -1.91798392779203
        struct[0].Gu[21,12] = -1.79198075607090
        struct[0].Gu[21,13] = 10.1957014483335
        struct[0].Gu[21,14] = 3.08738963687010
        struct[0].Gu[21,15] = 11.6035242966410
        struct[0].Gu[21,16] = -1.79198075607072
        struct[0].Gu[21,17] = 10.1957014483334
        struct[0].Gu[21,18] = -1.79181314887354
        struct[0].Gu[21,19] = 10.1955728673528
        struct[0].Gu[21,20] = 3.08755280951023
        struct[0].Gu[21,21] = 11.6033854030103
        struct[0].Gu[21,22] = -1.79181314887336
        struct[0].Gu[21,23] = 10.1955728673527
        struct[0].Gu[21,24] = -1.79131033552732
        struct[0].Gu[21,25] = 10.1951871226169
        struct[0].Gu[21,26] = 3.08804231916193
        struct[0].Gu[21,27] = 11.6029687203686
        struct[0].Gu[21,28] = -1.79131033552715
        struct[0].Gu[21,29] = 10.1951871226169
        struct[0].Gu[22,0] = -0.175091156146078
        struct[0].Gu[22,1] = -0.0000403160543779422
        struct[0].Gu[22,2] = -4.08928123428111E-17
        struct[0].Gu[22,3] = 3.10999136728748E-17
        struct[0].Gu[22,4] = 0.175091156146082
        struct[0].Gu[22,5] = 0.0000403160543753863
        struct[0].Gu[22,6] = -1.91798392779195
        struct[0].Gu[22,7] = 3.19497213887049
        struct[0].Gu[22,10] = 1.91798392779195
        struct[0].Gu[22,11] = -3.19497213887058
        struct[0].Gu[22,12] = 10.1957014483334
        struct[0].Gu[22,13] = 1.79198075607078
        struct[0].Gu[22,14] = 10.1957014483334
        struct[0].Gu[22,15] = 1.79198075607091
        struct[0].Gu[22,16] = 11.6035242966408
        struct[0].Gu[22,17] = -3.08738963687035
        struct[0].Gu[22,18] = 10.1955728673527
        struct[0].Gu[22,19] = 1.79181314887342
        struct[0].Gu[22,20] = 10.1955728673527
        struct[0].Gu[22,21] = 1.79181314887355
        struct[0].Gu[22,22] = 11.6033854030101
        struct[0].Gu[22,23] = -3.08755280951048
        struct[0].Gu[22,24] = 10.1951871226168
        struct[0].Gu[22,25] = 1.79131033552721
        struct[0].Gu[22,26] = 10.1951871226169
        struct[0].Gu[22,27] = 1.79131033552734
        struct[0].Gu[22,28] = 11.6029687203684
        struct[0].Gu[22,29] = -3.08804231916218
        struct[0].Gu[23,0] = 0.0000403160543779422
        struct[0].Gu[23,1] = -0.175091156146078
        struct[0].Gu[23,2] = -3.10999136728748E-17
        struct[0].Gu[23,3] = -4.08928123428111E-17
        struct[0].Gu[23,4] = -0.0000403160543753863
        struct[0].Gu[23,5] = 0.175091156146082
        struct[0].Gu[23,6] = -3.19497213887049
        struct[0].Gu[23,7] = -1.91798392779195
        struct[0].Gu[23,10] = 3.19497213887058
        struct[0].Gu[23,11] = 1.91798392779195
        struct[0].Gu[23,12] = -1.79198075607078
        struct[0].Gu[23,13] = 10.1957014483334
        struct[0].Gu[23,14] = -1.79198075607091
        struct[0].Gu[23,15] = 10.1957014483334
        struct[0].Gu[23,16] = 3.08738963687035
        struct[0].Gu[23,17] = 11.6035242966408
        struct[0].Gu[23,18] = -1.79181314887342
        struct[0].Gu[23,19] = 10.1955728673527
        struct[0].Gu[23,20] = -1.79181314887355
        struct[0].Gu[23,21] = 10.1955728673527
        struct[0].Gu[23,22] = 3.08755280951048
        struct[0].Gu[23,23] = 11.6033854030101
        struct[0].Gu[23,24] = -1.79131033552721
        struct[0].Gu[23,25] = 10.1951871226168
        struct[0].Gu[23,26] = -1.79131033552734
        struct[0].Gu[23,27] = 10.1951871226169
        struct[0].Gu[23,28] = 3.08804231916218
        struct[0].Gu[23,29] = 11.6029687203684
        struct[0].Gu[24,0] = 0.876104930717234
        struct[0].Gu[24,1] = -0.0328668071354569
        struct[0].Gu[24,2] = -0.124162742246183
        struct[0].Gu[24,3] = -0.0330297796399041
        struct[0].Gu[24,4] = -0.124162742246186
        struct[0].Gu[24,5] = -0.0330297796399051
        struct[0].Gu[24,6] = 8.99352976113163
        struct[0].Gu[24,7] = -16.3488065237228
        struct[0].Gu[24,8] = -1.96237550602395
        struct[0].Gu[24,9] = 1.90429395893588
        struct[0].Gu[24,10] = -1.96237550602400
        struct[0].Gu[24,11] = 1.90429395893593
        struct[0].Gu[24,12] = 1.91805727135914
        struct[0].Gu[24,13] = -3.19498294936924
        struct[0].Gu[24,16] = -1.91805727135928
        struct[0].Gu[24,17] = 3.19498294936902
        struct[0].Gu[24,18] = 1.91804912205592
        struct[0].Gu[24,19] = -3.19498174821903
        struct[0].Gu[24,22] = -1.91804912205606
        struct[0].Gu[24,23] = 3.19498174821881
        struct[0].Gu[24,24] = 1.91802467417320
        struct[0].Gu[24,25] = -3.19497814474392
        struct[0].Gu[24,28] = -1.91802467417334
        struct[0].Gu[24,29] = 3.19497814474371
        struct[0].Gu[25,0] = 0.0328668071354569
        struct[0].Gu[25,1] = 0.876104930717234
        struct[0].Gu[25,2] = 0.0330297796399041
        struct[0].Gu[25,3] = -0.124162742246183
        struct[0].Gu[25,4] = 0.0330297796399051
        struct[0].Gu[25,5] = -0.124162742246186
        struct[0].Gu[25,6] = 16.3488065237228
        struct[0].Gu[25,7] = 8.99352976113163
        struct[0].Gu[25,8] = -1.90429395893588
        struct[0].Gu[25,9] = -1.96237550602395
        struct[0].Gu[25,10] = -1.90429395893593
        struct[0].Gu[25,11] = -1.96237550602400
        struct[0].Gu[25,12] = 3.19498294936924
        struct[0].Gu[25,13] = 1.91805727135914
        struct[0].Gu[25,16] = -3.19498294936902
        struct[0].Gu[25,17] = -1.91805727135928
        struct[0].Gu[25,18] = 3.19498174821903
        struct[0].Gu[25,19] = 1.91804912205592
        struct[0].Gu[25,22] = -3.19498174821881
        struct[0].Gu[25,23] = -1.91804912205606
        struct[0].Gu[25,24] = 3.19497814474392
        struct[0].Gu[25,25] = 1.91802467417320
        struct[0].Gu[25,28] = -3.19497814474371
        struct[0].Gu[25,29] = -1.91802467417334
        struct[0].Gu[26,0] = -0.124162742246184
        struct[0].Gu[26,1] = -0.0330297796399032
        struct[0].Gu[26,2] = 0.876104930717237
        struct[0].Gu[26,3] = -0.0328668071354559
        struct[0].Gu[26,4] = -0.124162742246188
        struct[0].Gu[26,5] = -0.0330297796399070
        struct[0].Gu[26,6] = -1.96237550602395
        struct[0].Gu[26,7] = 1.90429395893591
        struct[0].Gu[26,8] = 8.99352976113168
        struct[0].Gu[26,9] = -16.3488065237228
        struct[0].Gu[26,10] = -1.96237550602406
        struct[0].Gu[26,11] = 1.90429395893593
        struct[0].Gu[26,12] = -1.91805727135909
        struct[0].Gu[26,13] = 3.19498294936934
        struct[0].Gu[26,14] = 1.91805727135939
        struct[0].Gu[26,15] = -3.19498294936900
        struct[0].Gu[26,18] = -1.91804912205587
        struct[0].Gu[26,19] = 3.19498174821913
        struct[0].Gu[26,20] = 1.91804912205617
        struct[0].Gu[26,21] = -3.19498174821879
        struct[0].Gu[26,24] = -1.91802467417315
        struct[0].Gu[26,25] = 3.19497814474403
        struct[0].Gu[26,26] = 1.91802467417345
        struct[0].Gu[26,27] = -3.19497814474369
        struct[0].Gu[27,0] = 0.0330297796399032
        struct[0].Gu[27,1] = -0.124162742246184
        struct[0].Gu[27,2] = 0.0328668071354559
        struct[0].Gu[27,3] = 0.876104930717237
        struct[0].Gu[27,4] = 0.0330297796399070
        struct[0].Gu[27,5] = -0.124162742246188
        struct[0].Gu[27,6] = -1.90429395893591
        struct[0].Gu[27,7] = -1.96237550602395
        struct[0].Gu[27,8] = 16.3488065237228
        struct[0].Gu[27,9] = 8.99352976113168
        struct[0].Gu[27,10] = -1.90429395893593
        struct[0].Gu[27,11] = -1.96237550602406
        struct[0].Gu[27,12] = -3.19498294936934
        struct[0].Gu[27,13] = -1.91805727135909
        struct[0].Gu[27,14] = 3.19498294936900
        struct[0].Gu[27,15] = 1.91805727135939
        struct[0].Gu[27,18] = -3.19498174821913
        struct[0].Gu[27,19] = -1.91804912205587
        struct[0].Gu[27,20] = 3.19498174821879
        struct[0].Gu[27,21] = 1.91804912205617
        struct[0].Gu[27,24] = -3.19497814474403
        struct[0].Gu[27,25] = -1.91802467417315
        struct[0].Gu[27,26] = 3.19497814474369
        struct[0].Gu[27,27] = 1.91802467417345
        struct[0].Gu[28,0] = -0.124162742246184
        struct[0].Gu[28,1] = -0.0330297796399061
        struct[0].Gu[28,2] = -0.124162742246189
        struct[0].Gu[28,3] = -0.0330297796399062
        struct[0].Gu[28,4] = 0.876104930717239
        struct[0].Gu[28,5] = -0.0328668071354540
        struct[0].Gu[28,6] = -1.96237550602401
        struct[0].Gu[28,7] = 1.90429395893589
        struct[0].Gu[28,8] = -1.96237550602406
        struct[0].Gu[28,9] = 1.90429395893597
        struct[0].Gu[28,10] = 8.99352976113174
        struct[0].Gu[28,11] = -16.3488065237228
        struct[0].Gu[28,14] = -1.91805727135935
        struct[0].Gu[28,15] = 3.19498294936915
        struct[0].Gu[28,16] = 1.91805727135921
        struct[0].Gu[28,17] = -3.19498294936923
        struct[0].Gu[28,20] = -1.91804912205613
        struct[0].Gu[28,21] = 3.19498174821894
        struct[0].Gu[28,22] = 1.91804912205598
        struct[0].Gu[28,23] = -3.19498174821902
        struct[0].Gu[28,26] = -1.91802467417341
        struct[0].Gu[28,27] = 3.19497814474383
        struct[0].Gu[28,28] = 1.91802467417326
        struct[0].Gu[28,29] = -3.19497814474392
        struct[0].Gu[29,0] = 0.0330297796399061
        struct[0].Gu[29,1] = -0.124162742246184
        struct[0].Gu[29,2] = 0.0330297796399062
        struct[0].Gu[29,3] = -0.124162742246189
        struct[0].Gu[29,4] = 0.0328668071354540
        struct[0].Gu[29,5] = 0.876104930717239
        struct[0].Gu[29,6] = -1.90429395893589
        struct[0].Gu[29,7] = -1.96237550602401
        struct[0].Gu[29,8] = -1.90429395893597
        struct[0].Gu[29,9] = -1.96237550602406
        struct[0].Gu[29,10] = 16.3488065237228
        struct[0].Gu[29,11] = 8.99352976113174
        struct[0].Gu[29,14] = -3.19498294936915
        struct[0].Gu[29,15] = -1.91805727135935
        struct[0].Gu[29,16] = 3.19498294936923
        struct[0].Gu[29,17] = 1.91805727135921
        struct[0].Gu[29,20] = -3.19498174821894
        struct[0].Gu[29,21] = -1.91804912205613
        struct[0].Gu[29,22] = 3.19498174821902
        struct[0].Gu[29,23] = 1.91804912205598
        struct[0].Gu[29,26] = -3.19497814474383
        struct[0].Gu[29,27] = -1.91802467417341
        struct[0].Gu[29,28] = 3.19497814474392
        struct[0].Gu[29,29] = 1.91802467417326
        struct[0].Gu[30,0] = 0.175093364713316
        struct[0].Gu[30,1] = 0.0000430097396370904
        struct[0].Gu[30,2] = -0.175093364713324
        struct[0].Gu[30,3] = -0.0000430097396321211
        struct[0].Gu[30,4] = 3.80502101367139E-17
        struct[0].Gu[30,5] = -3.95107998085596E-17
        struct[0].Gu[30,6] = 1.91805727135919
        struct[0].Gu[30,7] = -3.19498294936899
        struct[0].Gu[30,8] = -1.91805727135919
        struct[0].Gu[30,9] = 3.19498294936919
        struct[0].Gu[30,12] = 11.9248072396495
        struct[0].Gu[30,13] = -3.34841422289852
        struct[0].Gu[30,14] = 10.3248881664377
        struct[0].Gu[30,15] = 1.68849540047563
        struct[0].Gu[30,16] = 10.3248881664376
        struct[0].Gu[30,17] = 1.68849540047537
        struct[0].Gu[30,18] = 11.8179964829500
        struct[0].Gu[30,19] = -3.26107847080994
        struct[0].Gu[30,20] = 10.2820882609335
        struct[0].Gu[30,21] = 1.72332682544462
        struct[0].Gu[30,22] = 10.2820882609334
        struct[0].Gu[30,23] = 1.72332682544436
        struct[0].Gu[30,24] = 11.7109010138509
        struct[0].Gu[30,25] = -3.17407051442937
        struct[0].Gu[30,26] = 10.2390249864793
        struct[0].Gu[30,27] = 1.75782172888933
        struct[0].Gu[30,28] = 10.2390249864793
        struct[0].Gu[30,29] = 1.75782172888906
        struct[0].Gu[31,0] = -0.0000430097396370904
        struct[0].Gu[31,1] = 0.175093364713316
        struct[0].Gu[31,2] = 0.0000430097396321211
        struct[0].Gu[31,3] = -0.175093364713324
        struct[0].Gu[31,4] = 3.95107998085596E-17
        struct[0].Gu[31,5] = 3.80502101367139E-17
        struct[0].Gu[31,6] = 3.19498294936899
        struct[0].Gu[31,7] = 1.91805727135919
        struct[0].Gu[31,8] = -3.19498294936919
        struct[0].Gu[31,9] = -1.91805727135919
        struct[0].Gu[31,12] = 3.34841422289852
        struct[0].Gu[31,13] = 11.9248072396495
        struct[0].Gu[31,14] = -1.68849540047563
        struct[0].Gu[31,15] = 10.3248881664377
        struct[0].Gu[31,16] = -1.68849540047537
        struct[0].Gu[31,17] = 10.3248881664376
        struct[0].Gu[31,18] = 3.26107847080994
        struct[0].Gu[31,19] = 11.8179964829500
        struct[0].Gu[31,20] = -1.72332682544462
        struct[0].Gu[31,21] = 10.2820882609335
        struct[0].Gu[31,22] = -1.72332682544436
        struct[0].Gu[31,23] = 10.2820882609334
        struct[0].Gu[31,24] = 3.17407051442937
        struct[0].Gu[31,25] = 11.7109010138509
        struct[0].Gu[31,26] = -1.75782172888933
        struct[0].Gu[31,27] = 10.2390249864793
        struct[0].Gu[31,28] = -1.75782172888906
        struct[0].Gu[31,29] = 10.2390249864793
        struct[0].Gu[32,0] = 2.93583452294173E-18
        struct[0].Gu[32,1] = 4.89578301787291E-18
        struct[0].Gu[32,2] = 0.175093364713325
        struct[0].Gu[32,3] = 0.0000430097396379464
        struct[0].Gu[32,4] = -0.175093364713320
        struct[0].Gu[32,5] = -0.0000430097396419938
        struct[0].Gu[32,8] = 1.91805727135931
        struct[0].Gu[32,9] = -3.19498294936915
        struct[0].Gu[32,10] = -1.91805727135932
        struct[0].Gu[32,11] = 3.19498294936901
        struct[0].Gu[32,12] = 10.3248881664379
        struct[0].Gu[32,13] = 1.68849540047579
        struct[0].Gu[32,14] = 11.9248072396498
        struct[0].Gu[32,15] = -3.34841422289833
        struct[0].Gu[32,16] = 10.3248881664378
        struct[0].Gu[32,17] = 1.68849540047561
        struct[0].Gu[32,18] = 10.2820882609337
        struct[0].Gu[32,19] = 1.72332682544478
        struct[0].Gu[32,20] = 11.8179964829504
        struct[0].Gu[32,21] = -3.26107847080975
        struct[0].Gu[32,22] = 10.2820882609336
        struct[0].Gu[32,23] = 1.72332682544462
        struct[0].Gu[32,24] = 10.2390249864795
        struct[0].Gu[32,25] = 1.75782172888950
        struct[0].Gu[32,26] = 11.7109010138513
        struct[0].Gu[32,27] = -3.17407051442918
        struct[0].Gu[32,28] = 10.2390249864795
        struct[0].Gu[32,29] = 1.75782172888933
        struct[0].Gu[33,0] = -4.89578301787291E-18
        struct[0].Gu[33,1] = 2.93583452294173E-18
        struct[0].Gu[33,2] = -0.0000430097396379464
        struct[0].Gu[33,3] = 0.175093364713325
        struct[0].Gu[33,4] = 0.0000430097396419938
        struct[0].Gu[33,5] = -0.175093364713320
        struct[0].Gu[33,8] = 3.19498294936915
        struct[0].Gu[33,9] = 1.91805727135931
        struct[0].Gu[33,10] = -3.19498294936901
        struct[0].Gu[33,11] = -1.91805727135932
        struct[0].Gu[33,12] = -1.68849540047579
        struct[0].Gu[33,13] = 10.3248881664379
        struct[0].Gu[33,14] = 3.34841422289833
        struct[0].Gu[33,15] = 11.9248072396498
        struct[0].Gu[33,16] = -1.68849540047561
        struct[0].Gu[33,17] = 10.3248881664378
        struct[0].Gu[33,18] = -1.72332682544478
        struct[0].Gu[33,19] = 10.2820882609337
        struct[0].Gu[33,20] = 3.26107847080975
        struct[0].Gu[33,21] = 11.8179964829504
        struct[0].Gu[33,22] = -1.72332682544462
        struct[0].Gu[33,23] = 10.2820882609336
        struct[0].Gu[33,24] = -1.75782172888950
        struct[0].Gu[33,25] = 10.2390249864795
        struct[0].Gu[33,26] = 3.17407051442918
        struct[0].Gu[33,27] = 11.7109010138513
        struct[0].Gu[33,28] = -1.75782172888933
        struct[0].Gu[33,29] = 10.2390249864795
        struct[0].Gu[34,0] = -0.175093364713328
        struct[0].Gu[34,1] = -0.0000430097396325989
        struct[0].Gu[34,2] = -4.08938066674102E-17
        struct[0].Gu[34,3] = 3.10996770759589E-17
        struct[0].Gu[34,4] = 0.175093364713332
        struct[0].Gu[34,5] = 0.0000430097396300291
        struct[0].Gu[34,6] = -1.91805727135924
        struct[0].Gu[34,7] = 3.19498294936927
        struct[0].Gu[34,10] = 1.91805727135924
        struct[0].Gu[34,11] = -3.19498294936936
        struct[0].Gu[34,12] = 10.3248881664378
        struct[0].Gu[34,13] = 1.68849540047567
        struct[0].Gu[34,14] = 10.3248881664378
        struct[0].Gu[34,15] = 1.68849540047581
        struct[0].Gu[34,16] = 11.9248072396496
        struct[0].Gu[34,17] = -3.34841422289858
        struct[0].Gu[34,18] = 10.2820882609336
        struct[0].Gu[34,19] = 1.72332682544468
        struct[0].Gu[34,20] = 10.2820882609336
        struct[0].Gu[34,21] = 1.72332682544481
        struct[0].Gu[34,22] = 11.8179964829502
        struct[0].Gu[34,23] = -3.26107847081000
        struct[0].Gu[34,24] = 10.2390249864794
        struct[0].Gu[34,25] = 1.75782172888938
        struct[0].Gu[34,26] = 10.2390249864795
        struct[0].Gu[34,27] = 1.75782172888952
        struct[0].Gu[34,28] = 11.7109010138511
        struct[0].Gu[34,29] = -3.17407051442944
        struct[0].Gu[35,0] = 0.0000430097396325989
        struct[0].Gu[35,1] = -0.175093364713328
        struct[0].Gu[35,2] = -3.10996770759589E-17
        struct[0].Gu[35,3] = -4.08938066674102E-17
        struct[0].Gu[35,4] = -0.0000430097396300291
        struct[0].Gu[35,5] = 0.175093364713332
        struct[0].Gu[35,6] = -3.19498294936927
        struct[0].Gu[35,7] = -1.91805727135924
        struct[0].Gu[35,10] = 3.19498294936936
        struct[0].Gu[35,11] = 1.91805727135924
        struct[0].Gu[35,12] = -1.68849540047567
        struct[0].Gu[35,13] = 10.3248881664378
        struct[0].Gu[35,14] = -1.68849540047581
        struct[0].Gu[35,15] = 10.3248881664378
        struct[0].Gu[35,16] = 3.34841422289858
        struct[0].Gu[35,17] = 11.9248072396496
        struct[0].Gu[35,18] = -1.72332682544468
        struct[0].Gu[35,19] = 10.2820882609336
        struct[0].Gu[35,20] = -1.72332682544481
        struct[0].Gu[35,21] = 10.2820882609336
        struct[0].Gu[35,22] = 3.26107847081000
        struct[0].Gu[35,23] = 11.8179964829502
        struct[0].Gu[35,24] = -1.75782172888938
        struct[0].Gu[35,25] = 10.2390249864794
        struct[0].Gu[35,26] = -1.75782172888952
        struct[0].Gu[35,27] = 10.2390249864795
        struct[0].Gu[35,28] = 3.17407051442944
        struct[0].Gu[35,29] = 11.7109010138511
        struct[0].Gu[36,0] = 0.175093119317178
        struct[0].Gu[36,1] = 0.0000427104401568212
        struct[0].Gu[36,2] = -0.175093119317186
        struct[0].Gu[36,3] = -0.0000427104401518103
        struct[0].Gu[36,4] = 3.80500892677650E-17
        struct[0].Gu[36,5] = -3.95108094457718E-17
        struct[0].Gu[36,6] = 1.91804912205597
        struct[0].Gu[36,7] = -3.19498174821879
        struct[0].Gu[36,8] = -1.91804912205596
        struct[0].Gu[36,9] = 3.19498174821898
        struct[0].Gu[36,12] = 11.8179964829500
        struct[0].Gu[36,13] = -3.26107847080993
        struct[0].Gu[36,14] = 10.2820882609335
        struct[0].Gu[36,15] = 1.72332682544462
        struct[0].Gu[36,16] = 10.2820882609334
        struct[0].Gu[36,17] = 1.72332682544436
        struct[0].Gu[36,18] = 11.8178541267502
        struct[0].Gu[36,19] = -3.26124236866395
        struct[0].Gu[36,20] = 10.2819565764585
        struct[0].Gu[36,21] = 1.72315856468247
        struct[0].Gu[36,22] = 10.2819565764584
        struct[0].Gu[36,23] = 1.72315856468222
        struct[0].Gu[36,24] = 11.7107603897954
        struct[0].Gu[36,25] = -3.17423405384011
        struct[0].Gu[36,26] = 10.2388948546334
        struct[0].Gu[36,27] = 1.75765379075767
        struct[0].Gu[36,28] = 10.2388948546333
        struct[0].Gu[36,29] = 1.75765379075741
        struct[0].Gu[37,0] = -0.0000427104401568212
        struct[0].Gu[37,1] = 0.175093119317178
        struct[0].Gu[37,2] = 0.0000427104401518103
        struct[0].Gu[37,3] = -0.175093119317186
        struct[0].Gu[37,4] = 3.95108094457718E-17
        struct[0].Gu[37,5] = 3.80500892677650E-17
        struct[0].Gu[37,6] = 3.19498174821879
        struct[0].Gu[37,7] = 1.91804912205597
        struct[0].Gu[37,8] = -3.19498174821898
        struct[0].Gu[37,9] = -1.91804912205596
        struct[0].Gu[37,12] = 3.26107847080993
        struct[0].Gu[37,13] = 11.8179964829500
        struct[0].Gu[37,14] = -1.72332682544462
        struct[0].Gu[37,15] = 10.2820882609335
        struct[0].Gu[37,16] = -1.72332682544436
        struct[0].Gu[37,17] = 10.2820882609334
        struct[0].Gu[37,18] = 3.26124236866395
        struct[0].Gu[37,19] = 11.8178541267502
        struct[0].Gu[37,20] = -1.72315856468247
        struct[0].Gu[37,21] = 10.2819565764585
        struct[0].Gu[37,22] = -1.72315856468222
        struct[0].Gu[37,23] = 10.2819565764584
        struct[0].Gu[37,24] = 3.17423405384011
        struct[0].Gu[37,25] = 11.7107603897954
        struct[0].Gu[37,26] = -1.75765379075767
        struct[0].Gu[37,27] = 10.2388948546334
        struct[0].Gu[37,28] = -1.75765379075741
        struct[0].Gu[37,29] = 10.2388948546333
        struct[0].Gu[38,0] = 2.93583877411288E-18
        struct[0].Gu[38,1] = 4.89577113688110E-18
        struct[0].Gu[38,2] = 0.175093119317187
        struct[0].Gu[38,3] = 0.0000427104401575523
        struct[0].Gu[38,4] = -0.175093119317182
        struct[0].Gu[38,5] = -0.0000427104401616275
        struct[0].Gu[38,8] = 1.91804912205608
        struct[0].Gu[38,9] = -3.19498174821894
        struct[0].Gu[38,10] = -1.91804912205610
        struct[0].Gu[38,11] = 3.19498174821880
        struct[0].Gu[38,12] = 10.2820882609337
        struct[0].Gu[38,13] = 1.72332682544478
        struct[0].Gu[38,14] = 11.8179964829504
        struct[0].Gu[38,15] = -3.26107847080975
        struct[0].Gu[38,16] = 10.2820882609336
        struct[0].Gu[38,17] = 1.72332682544461
        struct[0].Gu[38,18] = 10.2819565764587
        struct[0].Gu[38,19] = 1.72315856468264
        struct[0].Gu[38,20] = 11.8178541267506
        struct[0].Gu[38,21] = -3.26124236866376
        struct[0].Gu[38,22] = 10.2819565764586
        struct[0].Gu[38,23] = 1.72315856468247
        struct[0].Gu[38,24] = 10.2388948546336
        struct[0].Gu[38,25] = 1.75765379075784
        struct[0].Gu[38,26] = 11.7107603897957
        struct[0].Gu[38,27] = -3.17423405383993
        struct[0].Gu[38,28] = 10.2388948546335
        struct[0].Gu[38,29] = 1.75765379075766
        struct[0].Gu[39,0] = -4.89577113688110E-18
        struct[0].Gu[39,1] = 2.93583877411288E-18
        struct[0].Gu[39,2] = -0.0000427104401575523
        struct[0].Gu[39,3] = 0.175093119317187
        struct[0].Gu[39,4] = 0.0000427104401616275
        struct[0].Gu[39,5] = -0.175093119317182
        struct[0].Gu[39,8] = 3.19498174821894
        struct[0].Gu[39,9] = 1.91804912205608
        struct[0].Gu[39,10] = -3.19498174821880
        struct[0].Gu[39,11] = -1.91804912205610
        struct[0].Gu[39,12] = -1.72332682544478
        struct[0].Gu[39,13] = 10.2820882609337
        struct[0].Gu[39,14] = 3.26107847080975
        struct[0].Gu[39,15] = 11.8179964829504
        struct[0].Gu[39,16] = -1.72332682544461
        struct[0].Gu[39,17] = 10.2820882609336
        struct[0].Gu[39,18] = -1.72315856468264
        struct[0].Gu[39,19] = 10.2819565764587
        struct[0].Gu[39,20] = 3.26124236866376
        struct[0].Gu[39,21] = 11.8178541267506
        struct[0].Gu[39,22] = -1.72315856468247
        struct[0].Gu[39,23] = 10.2819565764586
        struct[0].Gu[39,24] = -1.75765379075784
        struct[0].Gu[39,25] = 10.2388948546336
        struct[0].Gu[39,26] = 3.17423405383993
        struct[0].Gu[39,27] = 11.7107603897957
        struct[0].Gu[39,28] = -1.75765379075766
        struct[0].Gu[39,29] = 10.2388948546335
        struct[0].Gu[40,0] = -0.175093119317191
        struct[0].Gu[40,1] = -0.0000427104401522880
        struct[0].Gu[40,2] = -4.08936961867526E-17
        struct[0].Gu[40,3] = 3.10997033648060E-17
        struct[0].Gu[40,4] = 0.175093119317194
        struct[0].Gu[40,5] = 0.0000427104401497182
        struct[0].Gu[40,6] = -1.91804912205602
        struct[0].Gu[40,7] = 3.19498174821906
        struct[0].Gu[40,10] = 1.91804912205601
        struct[0].Gu[40,11] = -3.19498174821916
        struct[0].Gu[40,12] = 10.2820882609336
        struct[0].Gu[40,13] = 1.72332682544467
        struct[0].Gu[40,14] = 10.2820882609336
        struct[0].Gu[40,15] = 1.72332682544480
        struct[0].Gu[40,16] = 11.8179964829501
        struct[0].Gu[40,17] = -3.26107847081000
        struct[0].Gu[40,18] = 10.2819565764586
        struct[0].Gu[40,19] = 1.72315856468253
        struct[0].Gu[40,20] = 10.2819565764586
        struct[0].Gu[40,21] = 1.72315856468266
        struct[0].Gu[40,22] = 11.8178541267503
        struct[0].Gu[40,23] = -3.26124236866401
        struct[0].Gu[40,24] = 10.2388948546335
        struct[0].Gu[40,25] = 1.75765379075772
        struct[0].Gu[40,26] = 10.2388948546335
        struct[0].Gu[40,27] = 1.75765379075786
        struct[0].Gu[40,28] = 11.7107603897955
        struct[0].Gu[40,29] = -3.17423405384018
        struct[0].Gu[41,0] = 0.0000427104401522880
        struct[0].Gu[41,1] = -0.175093119317191
        struct[0].Gu[41,2] = -3.10997033648060E-17
        struct[0].Gu[41,3] = -4.08936961867526E-17
        struct[0].Gu[41,4] = -0.0000427104401497182
        struct[0].Gu[41,5] = 0.175093119317194
        struct[0].Gu[41,6] = -3.19498174821906
        struct[0].Gu[41,7] = -1.91804912205602
        struct[0].Gu[41,10] = 3.19498174821916
        struct[0].Gu[41,11] = 1.91804912205601
        struct[0].Gu[41,12] = -1.72332682544467
        struct[0].Gu[41,13] = 10.2820882609336
        struct[0].Gu[41,14] = -1.72332682544480
        struct[0].Gu[41,15] = 10.2820882609336
        struct[0].Gu[41,16] = 3.26107847081000
        struct[0].Gu[41,17] = 11.8179964829501
        struct[0].Gu[41,18] = -1.72315856468253
        struct[0].Gu[41,19] = 10.2819565764586
        struct[0].Gu[41,20] = -1.72315856468266
        struct[0].Gu[41,21] = 10.2819565764586
        struct[0].Gu[41,22] = 3.26124236866401
        struct[0].Gu[41,23] = 11.8178541267503
        struct[0].Gu[41,24] = -1.75765379075772
        struct[0].Gu[41,25] = 10.2388948546335
        struct[0].Gu[41,26] = -1.75765379075786
        struct[0].Gu[41,27] = 10.2388948546335
        struct[0].Gu[41,28] = 3.17423405384018
        struct[0].Gu[41,29] = 11.7107603897955
        struct[0].Gu[42,0] = 0.175092383128430
        struct[0].Gu[42,1] = 0.0000418125433939492
        struct[0].Gu[42,2] = -0.175092383128438
        struct[0].Gu[42,3] = -0.0000418125433889522
        struct[0].Gu[42,4] = 3.80497266612242E-17
        struct[0].Gu[42,5] = -3.95108383569680E-17
        struct[0].Gu[42,6] = 1.91802467417325
        struct[0].Gu[42,7] = -3.19497814474368
        struct[0].Gu[42,8] = -1.91802467417324
        struct[0].Gu[42,9] = 3.19497814474388
        struct[0].Gu[42,12] = 11.7109010138509
        struct[0].Gu[42,13] = -3.17407051442937
        struct[0].Gu[42,14] = 10.2390249864793
        struct[0].Gu[42,15] = 1.75782172888933
        struct[0].Gu[42,16] = 10.2390249864793
        struct[0].Gu[42,17] = 1.75782172888907
        struct[0].Gu[42,18] = 11.7107603897953
        struct[0].Gu[42,19] = -3.17423405384011
        struct[0].Gu[42,20] = 10.2388948546334
        struct[0].Gu[42,21] = 1.75765379075767
        struct[0].Gu[42,22] = 10.2388948546333
        struct[0].Gu[42,23] = 1.75765379075741
        struct[0].Gu[42,24] = 11.7103385159093
        struct[0].Gu[42,25] = -3.17472466374498
        struct[0].Gu[42,26] = 10.2385044573318
        struct[0].Gu[42,27] = 1.75714998466652
        struct[0].Gu[42,28] = 10.2385044573317
        struct[0].Gu[42,29] = 1.75714998466626
        struct[0].Gu[43,0] = -0.0000418125433939492
        struct[0].Gu[43,1] = 0.175092383128430
        struct[0].Gu[43,2] = 0.0000418125433889522
        struct[0].Gu[43,3] = -0.175092383128438
        struct[0].Gu[43,4] = 3.95108383569680E-17
        struct[0].Gu[43,5] = 3.80497266612242E-17
        struct[0].Gu[43,6] = 3.19497814474368
        struct[0].Gu[43,7] = 1.91802467417325
        struct[0].Gu[43,8] = -3.19497814474388
        struct[0].Gu[43,9] = -1.91802467417324
        struct[0].Gu[43,12] = 3.17407051442937
        struct[0].Gu[43,13] = 11.7109010138509
        struct[0].Gu[43,14] = -1.75782172888933
        struct[0].Gu[43,15] = 10.2390249864793
        struct[0].Gu[43,16] = -1.75782172888907
        struct[0].Gu[43,17] = 10.2390249864793
        struct[0].Gu[43,18] = 3.17423405384011
        struct[0].Gu[43,19] = 11.7107603897953
        struct[0].Gu[43,20] = -1.75765379075767
        struct[0].Gu[43,21] = 10.2388948546334
        struct[0].Gu[43,22] = -1.75765379075741
        struct[0].Gu[43,23] = 10.2388948546333
        struct[0].Gu[43,24] = 3.17472466374498
        struct[0].Gu[43,25] = 11.7103385159093
        struct[0].Gu[43,26] = -1.75714998466652
        struct[0].Gu[43,27] = 10.2385044573318
        struct[0].Gu[43,28] = -1.75714998466626
        struct[0].Gu[43,29] = 10.2385044573317
        struct[0].Gu[44,0] = 2.93585152757382E-18
        struct[0].Gu[44,1] = 4.89573549392443E-18
        struct[0].Gu[44,2] = 0.175092383128439
        struct[0].Gu[44,3] = 0.0000418125433946109
        struct[0].Gu[44,4] = -0.175092383128434
        struct[0].Gu[44,5] = -0.0000418125433986861
        struct[0].Gu[44,8] = 1.91802467417336
        struct[0].Gu[44,9] = -3.19497814474384
        struct[0].Gu[44,10] = -1.91802467417338
        struct[0].Gu[44,11] = 3.19497814474370
        struct[0].Gu[44,12] = 10.2390249864795
        struct[0].Gu[44,13] = 1.75782172888950
        struct[0].Gu[44,14] = 11.7109010138513
        struct[0].Gu[44,15] = -3.17407051442919
        struct[0].Gu[44,16] = 10.2390249864794
        struct[0].Gu[44,17] = 1.75782172888932
        struct[0].Gu[44,18] = 10.2388948546336
        struct[0].Gu[44,19] = 1.75765379075784
        struct[0].Gu[44,20] = 11.7107603897957
        struct[0].Gu[44,21] = -3.17423405383993
        struct[0].Gu[44,22] = 10.2388948546335
        struct[0].Gu[44,23] = 1.75765379075766
        struct[0].Gu[44,24] = 10.2385044573320
        struct[0].Gu[44,25] = 1.75714998466669
        struct[0].Gu[44,26] = 11.7103385159096
        struct[0].Gu[44,27] = -3.17472466374480
        struct[0].Gu[44,28] = 10.2385044573319
        struct[0].Gu[44,29] = 1.75714998466651
        struct[0].Gu[45,0] = -4.89573549392443E-18
        struct[0].Gu[45,1] = 2.93585152757382E-18
        struct[0].Gu[45,2] = -0.0000418125433946109
        struct[0].Gu[45,3] = 0.175092383128439
        struct[0].Gu[45,4] = 0.0000418125433986861
        struct[0].Gu[45,5] = -0.175092383128434
        struct[0].Gu[45,8] = 3.19497814474384
        struct[0].Gu[45,9] = 1.91802467417336
        struct[0].Gu[45,10] = -3.19497814474370
        struct[0].Gu[45,11] = -1.91802467417338
        struct[0].Gu[45,12] = -1.75782172888950
        struct[0].Gu[45,13] = 10.2390249864795
        struct[0].Gu[45,14] = 3.17407051442919
        struct[0].Gu[45,15] = 11.7109010138513
        struct[0].Gu[45,16] = -1.75782172888932
        struct[0].Gu[45,17] = 10.2390249864794
        struct[0].Gu[45,18] = -1.75765379075784
        struct[0].Gu[45,19] = 10.2388948546336
        struct[0].Gu[45,20] = 3.17423405383993
        struct[0].Gu[45,21] = 11.7107603897957
        struct[0].Gu[45,22] = -1.75765379075766
        struct[0].Gu[45,23] = 10.2388948546335
        struct[0].Gu[45,24] = -1.75714998466669
        struct[0].Gu[45,25] = 10.2385044573320
        struct[0].Gu[45,26] = 3.17472466374480
        struct[0].Gu[45,27] = 11.7103385159096
        struct[0].Gu[45,28] = -1.75714998466651
        struct[0].Gu[45,29] = 10.2385044573319
        struct[0].Gu[46,0] = -0.175092383128442
        struct[0].Gu[46,1] = -0.0000418125433894160
        struct[0].Gu[46,2] = -4.08933647449996E-17
        struct[0].Gu[46,3] = 3.10997822308960E-17
        struct[0].Gu[46,4] = 0.175092383128446
        struct[0].Gu[46,5] = 0.0000418125433868462
        struct[0].Gu[46,6] = -1.91802467417330
        struct[0].Gu[46,7] = 3.19497814474395
        struct[0].Gu[46,10] = 1.91802467417329
        struct[0].Gu[46,11] = -3.19497814474405
        struct[0].Gu[46,12] = 10.2390249864794
        struct[0].Gu[46,13] = 1.75782172888938
        struct[0].Gu[46,14] = 10.2390249864795
        struct[0].Gu[46,15] = 1.75782172888951
        struct[0].Gu[46,16] = 11.7109010138511
        struct[0].Gu[46,17] = -3.17407051442944
        struct[0].Gu[46,18] = 10.2388948546335
        struct[0].Gu[46,19] = 1.75765379075772
        struct[0].Gu[46,20] = 10.2388948546335
        struct[0].Gu[46,21] = 1.75765379075785
        struct[0].Gu[46,22] = 11.7107603897955
        struct[0].Gu[46,23] = -3.17423405384018
        struct[0].Gu[46,24] = 10.2385044573319
        struct[0].Gu[46,25] = 1.75714998466657
        struct[0].Gu[46,26] = 10.2385044573319
        struct[0].Gu[46,27] = 1.75714998466670
        struct[0].Gu[46,28] = 11.7103385159094
        struct[0].Gu[46,29] = -3.17472466374505
        struct[0].Gu[47,0] = 0.0000418125433894160
        struct[0].Gu[47,1] = -0.175092383128442
        struct[0].Gu[47,2] = -3.10997822308960E-17
        struct[0].Gu[47,3] = -4.08933647449996E-17
        struct[0].Gu[47,4] = -0.0000418125433868462
        struct[0].Gu[47,5] = 0.175092383128446
        struct[0].Gu[47,6] = -3.19497814474395
        struct[0].Gu[47,7] = -1.91802467417330
        struct[0].Gu[47,10] = 3.19497814474405
        struct[0].Gu[47,11] = 1.91802467417329
        struct[0].Gu[47,12] = -1.75782172888938
        struct[0].Gu[47,13] = 10.2390249864794
        struct[0].Gu[47,14] = -1.75782172888951
        struct[0].Gu[47,15] = 10.2390249864795
        struct[0].Gu[47,16] = 3.17407051442944
        struct[0].Gu[47,17] = 11.7109010138511
        struct[0].Gu[47,18] = -1.75765379075772
        struct[0].Gu[47,19] = 10.2388948546335
        struct[0].Gu[47,20] = -1.75765379075785
        struct[0].Gu[47,21] = 10.2388948546335
        struct[0].Gu[47,22] = 3.17423405384018
        struct[0].Gu[47,23] = 11.7107603897955
        struct[0].Gu[47,24] = -1.75714998466657
        struct[0].Gu[47,25] = 10.2385044573319
        struct[0].Gu[47,26] = -1.75714998466670
        struct[0].Gu[47,27] = 10.2385044573319
        struct[0].Gu[47,28] = 3.17472466374505
        struct[0].Gu[47,29] = 11.7103385159094


        struct[0].Hy[3,0] = 1.0*v_W1lv_a_r*(v_W1lv_a_i**2 + v_W1lv_a_r**2)**(-0.5)
        struct[0].Hy[3,1] = 1.0*v_W1lv_a_i*(v_W1lv_a_i**2 + v_W1lv_a_r**2)**(-0.5)
        struct[0].Hy[4,2] = 1.0*v_W1lv_b_r*(v_W1lv_b_i**2 + v_W1lv_b_r**2)**(-0.5)
        struct[0].Hy[4,3] = 1.0*v_W1lv_b_i*(v_W1lv_b_i**2 + v_W1lv_b_r**2)**(-0.5)
        struct[0].Hy[5,4] = 1.0*v_W1lv_c_r*(v_W1lv_c_i**2 + v_W1lv_c_r**2)**(-0.5)
        struct[0].Hy[5,5] = 1.0*v_W1lv_c_i*(v_W1lv_c_i**2 + v_W1lv_c_r**2)**(-0.5)
        struct[0].Hy[6,6] = 1.0*v_W2lv_a_r*(v_W2lv_a_i**2 + v_W2lv_a_r**2)**(-0.5)
        struct[0].Hy[6,7] = 1.0*v_W2lv_a_i*(v_W2lv_a_i**2 + v_W2lv_a_r**2)**(-0.5)
        struct[0].Hy[7,8] = 1.0*v_W2lv_b_r*(v_W2lv_b_i**2 + v_W2lv_b_r**2)**(-0.5)
        struct[0].Hy[7,9] = 1.0*v_W2lv_b_i*(v_W2lv_b_i**2 + v_W2lv_b_r**2)**(-0.5)
        struct[0].Hy[8,10] = 1.0*v_W2lv_c_r*(v_W2lv_c_i**2 + v_W2lv_c_r**2)**(-0.5)
        struct[0].Hy[8,11] = 1.0*v_W2lv_c_i*(v_W2lv_c_i**2 + v_W2lv_c_r**2)**(-0.5)
        struct[0].Hy[9,12] = 1.0*v_W3lv_a_r*(v_W3lv_a_i**2 + v_W3lv_a_r**2)**(-0.5)
        struct[0].Hy[9,13] = 1.0*v_W3lv_a_i*(v_W3lv_a_i**2 + v_W3lv_a_r**2)**(-0.5)
        struct[0].Hy[10,14] = 1.0*v_W3lv_b_r*(v_W3lv_b_i**2 + v_W3lv_b_r**2)**(-0.5)
        struct[0].Hy[10,15] = 1.0*v_W3lv_b_i*(v_W3lv_b_i**2 + v_W3lv_b_r**2)**(-0.5)
        struct[0].Hy[11,16] = 1.0*v_W3lv_c_r*(v_W3lv_c_i**2 + v_W3lv_c_r**2)**(-0.5)
        struct[0].Hy[11,17] = 1.0*v_W3lv_c_i*(v_W3lv_c_i**2 + v_W3lv_c_r**2)**(-0.5)
        struct[0].Hy[12,18] = 1.0*v_POImv_a_r*(v_POImv_a_i**2 + v_POImv_a_r**2)**(-0.5)
        struct[0].Hy[12,19] = 1.0*v_POImv_a_i*(v_POImv_a_i**2 + v_POImv_a_r**2)**(-0.5)
        struct[0].Hy[13,20] = 1.0*v_POImv_b_r*(v_POImv_b_i**2 + v_POImv_b_r**2)**(-0.5)
        struct[0].Hy[13,21] = 1.0*v_POImv_b_i*(v_POImv_b_i**2 + v_POImv_b_r**2)**(-0.5)
        struct[0].Hy[14,22] = 1.0*v_POImv_c_r*(v_POImv_c_i**2 + v_POImv_c_r**2)**(-0.5)
        struct[0].Hy[14,23] = 1.0*v_POImv_c_i*(v_POImv_c_i**2 + v_POImv_c_r**2)**(-0.5)
        struct[0].Hy[15,24] = 1.0*v_POI_a_r*(v_POI_a_i**2 + v_POI_a_r**2)**(-0.5)
        struct[0].Hy[15,25] = 1.0*v_POI_a_i*(v_POI_a_i**2 + v_POI_a_r**2)**(-0.5)
        struct[0].Hy[16,26] = 1.0*v_POI_b_r*(v_POI_b_i**2 + v_POI_b_r**2)**(-0.5)
        struct[0].Hy[16,27] = 1.0*v_POI_b_i*(v_POI_b_i**2 + v_POI_b_r**2)**(-0.5)
        struct[0].Hy[17,28] = 1.0*v_POI_c_r*(v_POI_c_i**2 + v_POI_c_r**2)**(-0.5)
        struct[0].Hy[17,29] = 1.0*v_POI_c_i*(v_POI_c_i**2 + v_POI_c_r**2)**(-0.5)
        struct[0].Hy[18,30] = 1.0*v_W1mv_a_r*(v_W1mv_a_i**2 + v_W1mv_a_r**2)**(-0.5)
        struct[0].Hy[18,31] = 1.0*v_W1mv_a_i*(v_W1mv_a_i**2 + v_W1mv_a_r**2)**(-0.5)
        struct[0].Hy[19,32] = 1.0*v_W1mv_b_r*(v_W1mv_b_i**2 + v_W1mv_b_r**2)**(-0.5)
        struct[0].Hy[19,33] = 1.0*v_W1mv_b_i*(v_W1mv_b_i**2 + v_W1mv_b_r**2)**(-0.5)
        struct[0].Hy[20,34] = 1.0*v_W1mv_c_r*(v_W1mv_c_i**2 + v_W1mv_c_r**2)**(-0.5)
        struct[0].Hy[20,35] = 1.0*v_W1mv_c_i*(v_W1mv_c_i**2 + v_W1mv_c_r**2)**(-0.5)
        struct[0].Hy[21,36] = 1.0*v_W2mv_a_r*(v_W2mv_a_i**2 + v_W2mv_a_r**2)**(-0.5)
        struct[0].Hy[21,37] = 1.0*v_W2mv_a_i*(v_W2mv_a_i**2 + v_W2mv_a_r**2)**(-0.5)
        struct[0].Hy[22,38] = 1.0*v_W2mv_b_r*(v_W2mv_b_i**2 + v_W2mv_b_r**2)**(-0.5)
        struct[0].Hy[22,39] = 1.0*v_W2mv_b_i*(v_W2mv_b_i**2 + v_W2mv_b_r**2)**(-0.5)
        struct[0].Hy[23,40] = 1.0*v_W2mv_c_r*(v_W2mv_c_i**2 + v_W2mv_c_r**2)**(-0.5)
        struct[0].Hy[23,41] = 1.0*v_W2mv_c_i*(v_W2mv_c_i**2 + v_W2mv_c_r**2)**(-0.5)
        struct[0].Hy[24,42] = 1.0*v_W3mv_a_r*(v_W3mv_a_i**2 + v_W3mv_a_r**2)**(-0.5)
        struct[0].Hy[24,43] = 1.0*v_W3mv_a_i*(v_W3mv_a_i**2 + v_W3mv_a_r**2)**(-0.5)
        struct[0].Hy[25,44] = 1.0*v_W3mv_b_r*(v_W3mv_b_i**2 + v_W3mv_b_r**2)**(-0.5)
        struct[0].Hy[25,45] = 1.0*v_W3mv_b_i*(v_W3mv_b_i**2 + v_W3mv_b_r**2)**(-0.5)
        struct[0].Hy[26,46] = 1.0*v_W3mv_c_r*(v_W3mv_c_i**2 + v_W3mv_c_r**2)**(-0.5)
        struct[0].Hy[26,47] = 1.0*v_W3mv_c_i*(v_W3mv_c_i**2 + v_W3mv_c_r**2)**(-0.5)

        struct[0].Hu[0,0] = 1.0*v_GRID_a_r*(v_GRID_a_i**2 + v_GRID_a_r**2)**(-0.5)
        struct[0].Hu[0,1] = 1.0*v_GRID_a_i*(v_GRID_a_i**2 + v_GRID_a_r**2)**(-0.5)
        struct[0].Hu[1,2] = 1.0*v_GRID_b_r*(v_GRID_b_i**2 + v_GRID_b_r**2)**(-0.5)
        struct[0].Hu[1,3] = 1.0*v_GRID_b_i*(v_GRID_b_i**2 + v_GRID_b_r**2)**(-0.5)
        struct[0].Hu[2,4] = 1.0*v_GRID_c_r*(v_GRID_c_i**2 + v_GRID_c_r**2)**(-0.5)
        struct[0].Hu[2,5] = 1.0*v_GRID_c_i*(v_GRID_c_i**2 + v_GRID_c_r**2)**(-0.5)



@numba.njit(cache=True)
def ini(struct,mode):

    # Parameters:
    
    # Inputs:
    v_GRID_a_r = struct[0].v_GRID_a_r
    v_GRID_a_i = struct[0].v_GRID_a_i
    v_GRID_b_r = struct[0].v_GRID_b_r
    v_GRID_b_i = struct[0].v_GRID_b_i
    v_GRID_c_r = struct[0].v_GRID_c_r
    v_GRID_c_i = struct[0].v_GRID_c_i
    i_POI_a_r = struct[0].i_POI_a_r
    i_POI_a_i = struct[0].i_POI_a_i
    i_POI_b_r = struct[0].i_POI_b_r
    i_POI_b_i = struct[0].i_POI_b_i
    i_POI_c_r = struct[0].i_POI_c_r
    i_POI_c_i = struct[0].i_POI_c_i
    i_W1mv_a_r = struct[0].i_W1mv_a_r
    i_W1mv_a_i = struct[0].i_W1mv_a_i
    i_W1mv_b_r = struct[0].i_W1mv_b_r
    i_W1mv_b_i = struct[0].i_W1mv_b_i
    i_W1mv_c_r = struct[0].i_W1mv_c_r
    i_W1mv_c_i = struct[0].i_W1mv_c_i
    i_W2mv_a_r = struct[0].i_W2mv_a_r
    i_W2mv_a_i = struct[0].i_W2mv_a_i
    i_W2mv_b_r = struct[0].i_W2mv_b_r
    i_W2mv_b_i = struct[0].i_W2mv_b_i
    i_W2mv_c_r = struct[0].i_W2mv_c_r
    i_W2mv_c_i = struct[0].i_W2mv_c_i
    i_W3mv_a_r = struct[0].i_W3mv_a_r
    i_W3mv_a_i = struct[0].i_W3mv_a_i
    i_W3mv_b_r = struct[0].i_W3mv_b_r
    i_W3mv_b_i = struct[0].i_W3mv_b_i
    i_W3mv_c_r = struct[0].i_W3mv_c_r
    i_W3mv_c_i = struct[0].i_W3mv_c_i
    p_W1lv_a = struct[0].p_W1lv_a
    q_W1lv_a = struct[0].q_W1lv_a
    p_W1lv_b = struct[0].p_W1lv_b
    q_W1lv_b = struct[0].q_W1lv_b
    p_W1lv_c = struct[0].p_W1lv_c
    q_W1lv_c = struct[0].q_W1lv_c
    p_W2lv_a = struct[0].p_W2lv_a
    q_W2lv_a = struct[0].q_W2lv_a
    p_W2lv_b = struct[0].p_W2lv_b
    q_W2lv_b = struct[0].q_W2lv_b
    p_W2lv_c = struct[0].p_W2lv_c
    q_W2lv_c = struct[0].q_W2lv_c
    p_W3lv_a = struct[0].p_W3lv_a
    q_W3lv_a = struct[0].q_W3lv_a
    p_W3lv_b = struct[0].p_W3lv_b
    q_W3lv_b = struct[0].q_W3lv_b
    p_W3lv_c = struct[0].p_W3lv_c
    q_W3lv_c = struct[0].q_W3lv_c
    p_POImv_a = struct[0].p_POImv_a
    q_POImv_a = struct[0].q_POImv_a
    p_POImv_b = struct[0].p_POImv_b
    q_POImv_b = struct[0].q_POImv_b
    p_POImv_c = struct[0].p_POImv_c
    q_POImv_c = struct[0].q_POImv_c
    
    # Dynamical states:
    a = struct[0].x[0,0]
    
    # Algebraic states:
    v_W1lv_a_r = struct[0].y_ini[0,0]
    v_W1lv_a_i = struct[0].y_ini[1,0]
    v_W1lv_b_r = struct[0].y_ini[2,0]
    v_W1lv_b_i = struct[0].y_ini[3,0]
    v_W1lv_c_r = struct[0].y_ini[4,0]
    v_W1lv_c_i = struct[0].y_ini[5,0]
    v_W2lv_a_r = struct[0].y_ini[6,0]
    v_W2lv_a_i = struct[0].y_ini[7,0]
    v_W2lv_b_r = struct[0].y_ini[8,0]
    v_W2lv_b_i = struct[0].y_ini[9,0]
    v_W2lv_c_r = struct[0].y_ini[10,0]
    v_W2lv_c_i = struct[0].y_ini[11,0]
    v_W3lv_a_r = struct[0].y_ini[12,0]
    v_W3lv_a_i = struct[0].y_ini[13,0]
    v_W3lv_b_r = struct[0].y_ini[14,0]
    v_W3lv_b_i = struct[0].y_ini[15,0]
    v_W3lv_c_r = struct[0].y_ini[16,0]
    v_W3lv_c_i = struct[0].y_ini[17,0]
    v_POImv_a_r = struct[0].y_ini[18,0]
    v_POImv_a_i = struct[0].y_ini[19,0]
    v_POImv_b_r = struct[0].y_ini[20,0]
    v_POImv_b_i = struct[0].y_ini[21,0]
    v_POImv_c_r = struct[0].y_ini[22,0]
    v_POImv_c_i = struct[0].y_ini[23,0]
    v_POI_a_r = struct[0].y_ini[24,0]
    v_POI_a_i = struct[0].y_ini[25,0]
    v_POI_b_r = struct[0].y_ini[26,0]
    v_POI_b_i = struct[0].y_ini[27,0]
    v_POI_c_r = struct[0].y_ini[28,0]
    v_POI_c_i = struct[0].y_ini[29,0]
    v_W1mv_a_r = struct[0].y_ini[30,0]
    v_W1mv_a_i = struct[0].y_ini[31,0]
    v_W1mv_b_r = struct[0].y_ini[32,0]
    v_W1mv_b_i = struct[0].y_ini[33,0]
    v_W1mv_c_r = struct[0].y_ini[34,0]
    v_W1mv_c_i = struct[0].y_ini[35,0]
    v_W2mv_a_r = struct[0].y_ini[36,0]
    v_W2mv_a_i = struct[0].y_ini[37,0]
    v_W2mv_b_r = struct[0].y_ini[38,0]
    v_W2mv_b_i = struct[0].y_ini[39,0]
    v_W2mv_c_r = struct[0].y_ini[40,0]
    v_W2mv_c_i = struct[0].y_ini[41,0]
    v_W3mv_a_r = struct[0].y_ini[42,0]
    v_W3mv_a_i = struct[0].y_ini[43,0]
    v_W3mv_b_r = struct[0].y_ini[44,0]
    v_W3mv_b_i = struct[0].y_ini[45,0]
    v_W3mv_c_r = struct[0].y_ini[46,0]
    v_W3mv_c_i = struct[0].y_ini[47,0]
    i_W1lv_a_r = struct[0].y_ini[48,0]
    i_W1lv_a_i = struct[0].y_ini[49,0]
    i_W1lv_b_r = struct[0].y_ini[50,0]
    i_W1lv_b_i = struct[0].y_ini[51,0]
    i_W1lv_c_r = struct[0].y_ini[52,0]
    i_W1lv_c_i = struct[0].y_ini[53,0]
    i_W2lv_a_r = struct[0].y_ini[54,0]
    i_W2lv_a_i = struct[0].y_ini[55,0]
    i_W2lv_b_r = struct[0].y_ini[56,0]
    i_W2lv_b_i = struct[0].y_ini[57,0]
    i_W2lv_c_r = struct[0].y_ini[58,0]
    i_W2lv_c_i = struct[0].y_ini[59,0]
    i_W3lv_a_r = struct[0].y_ini[60,0]
    i_W3lv_a_i = struct[0].y_ini[61,0]
    i_W3lv_b_r = struct[0].y_ini[62,0]
    i_W3lv_b_i = struct[0].y_ini[63,0]
    i_W3lv_c_r = struct[0].y_ini[64,0]
    i_W3lv_c_i = struct[0].y_ini[65,0]
    i_POImv_a_r = struct[0].y_ini[66,0]
    i_POImv_a_i = struct[0].y_ini[67,0]
    i_POImv_b_r = struct[0].y_ini[68,0]
    i_POImv_b_i = struct[0].y_ini[69,0]
    i_POImv_c_r = struct[0].y_ini[70,0]
    i_POImv_c_i = struct[0].y_ini[71,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = 1 - a
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -0.127279074345343*i_POI_a_i + 0.076409970853885*i_POI_a_r + 0.0636395371726705*i_POI_b_i - 0.038204985426942*i_POI_b_r + 0.0636395371726721*i_POI_c_i - 0.0382049854269431*i_POI_c_r - 0.0971901504394946*i_POImv_a_i + 0.0280418380652321*i_POImv_a_r + 0.0971901504394895*i_POImv_c_i - 0.0280418380652375*i_POImv_c_r - 0.0154231877861473*i_W1lv_a_i + 0.0031739357845936*i_W1lv_a_r + 0.00199839389307364*i_W1lv_b_i - 0.000634767892296793*i_W1lv_b_r + 0.00199839389307368*i_W1lv_c_i - 0.000634767892296814*i_W1lv_c_r - 0.100328108879392*i_W1mv_a_i + 0.0318681229122137*i_W1mv_a_r + 0.100328108879387*i_W1mv_c_i - 0.0318681229122191*i_W1mv_c_r - 0.00395512560257793*i_W2lv_a_i + 0.00121874317417018*i_W2lv_a_r + 0.00197756280128894*i_W2lv_b_i - 0.00060937158708508*i_W2lv_b_r + 0.00197756280128899*i_W2lv_c_i - 0.000609371587085102*i_W2lv_c_r - 0.0992822970142266*i_W2mv_a_i + 0.0305931173773959*i_W2mv_a_r + 0.0992822970142214*i_W2mv_c_i - 0.0305931173774017*i_W2mv_c_r - 0.00391345649507333*i_W3lv_a_i + 0.00116793362771941*i_W3lv_a_r + 0.00195672824753664*i_W3lv_b_i - 0.000583966813859693*i_W3lv_b_r + 0.00195672824753669*i_W3lv_c_i - 0.000583966813859714*i_W3lv_c_r - 0.0982363113431537*i_W3mv_a_i + 0.0293176867112763*i_W3mv_a_r + 0.0982363113431486*i_W3mv_c_i - 0.0293176867112818*i_W3mv_c_r + 1.71338624816169e-6*v_GRID_a_i + 0.00697522388628822*v_GRID_a_r - 8.56693124078082e-7*v_GRID_b_i - 0.00348761194314407*v_GRID_b_r - 8.56693124085475e-7*v_GRID_c_i - 0.00348761194314416*v_GRID_c_r - v_W1lv_a_r
        struct[0].g[1,0] = 0.076409970853885*i_POI_a_i + 0.127279074345343*i_POI_a_r - 0.038204985426942*i_POI_b_i - 0.0636395371726705*i_POI_b_r - 0.0382049854269431*i_POI_c_i - 0.0636395371726721*i_POI_c_r + 0.0280418380652321*i_POImv_a_i + 0.0971901504394946*i_POImv_a_r - 0.0280418380652375*i_POImv_c_i - 0.0971901504394895*i_POImv_c_r + 0.0031739357845936*i_W1lv_a_i + 0.0154231877861473*i_W1lv_a_r - 0.000634767892296793*i_W1lv_b_i - 0.00199839389307364*i_W1lv_b_r - 0.000634767892296814*i_W1lv_c_i - 0.00199839389307368*i_W1lv_c_r + 0.0318681229122137*i_W1mv_a_i + 0.100328108879392*i_W1mv_a_r - 0.0318681229122191*i_W1mv_c_i - 0.100328108879387*i_W1mv_c_r + 0.00121874317417018*i_W2lv_a_i + 0.00395512560257793*i_W2lv_a_r - 0.00060937158708508*i_W2lv_b_i - 0.00197756280128894*i_W2lv_b_r - 0.000609371587085102*i_W2lv_c_i - 0.00197756280128899*i_W2lv_c_r + 0.0305931173773959*i_W2mv_a_i + 0.0992822970142266*i_W2mv_a_r - 0.0305931173774017*i_W2mv_c_i - 0.0992822970142214*i_W2mv_c_r + 0.00116793362771941*i_W3lv_a_i + 0.00391345649507333*i_W3lv_a_r - 0.000583966813859693*i_W3lv_b_i - 0.00195672824753664*i_W3lv_b_r - 0.000583966813859714*i_W3lv_c_i - 0.00195672824753669*i_W3lv_c_r + 0.0293176867112763*i_W3mv_a_i + 0.0982363113431537*i_W3mv_a_r - 0.0293176867112818*i_W3mv_c_i - 0.0982363113431486*i_W3mv_c_r + 0.00697522388628822*v_GRID_a_i - 1.71338624816169e-6*v_GRID_a_r - 0.00348761194314407*v_GRID_b_i + 8.56693124078082e-7*v_GRID_b_r - 0.00348761194314416*v_GRID_c_i + 8.56693124085475e-7*v_GRID_c_r - v_W1lv_a_i
        struct[0].g[2,0] = 0.0636395371726714*i_POI_a_i - 0.0382049854269416*i_POI_a_r - 0.127279074345344*i_POI_b_i + 0.0764099708538862*i_POI_b_r + 0.0636395371726728*i_POI_c_i - 0.0382049854269445*i_POI_c_r + 0.0971901504394967*i_POImv_a_i - 0.0280418380652299*i_POImv_a_r - 0.0971901504394895*i_POImv_b_i + 0.0280418380652406*i_POImv_b_r + 0.00199839389307366*i_W1lv_a_i - 0.000634767892296777*i_W1lv_a_r - 0.0154231877861474*i_W1lv_b_i + 0.00317393578459362*i_W1lv_b_r + 0.00199839389307372*i_W1lv_c_i - 0.000634767892296846*i_W1lv_c_r + 0.100328108879395*i_W1mv_a_i - 0.0318681229122115*i_W1mv_a_r - 0.100328108879388*i_W1mv_b_i + 0.0318681229122223*i_W1mv_b_r + 0.00197756280128896*i_W2lv_a_i - 0.000609371587085066*i_W2lv_a_r - 0.00395512560257798*i_W2lv_b_i + 0.0012187431741702*i_W2lv_b_r + 0.00197756280128902*i_W2lv_c_i - 0.000609371587085134*i_W2lv_c_r + 0.0992822970142287*i_W2mv_a_i - 0.0305931173773937*i_W2mv_a_r - 0.0992822970142215*i_W2mv_b_i + 0.0305931173774048*i_W2mv_b_r + 0.00195672824753666*i_W3lv_a_i - 0.000583966813859679*i_W3lv_a_r - 0.00391345649507338*i_W3lv_b_i + 0.00116793362771943*i_W3lv_b_r + 0.00195672824753672*i_W3lv_c_i - 0.000583966813859747*i_W3lv_c_r + 0.098236311343156*i_W3mv_a_i - 0.0293176867112742*i_W3mv_a_r - 0.0982363113431489*i_W3mv_b_i + 0.029317686711285*i_W3mv_b_r - 8.5669312404275e-7*v_GRID_a_i - 0.00348761194314409*v_GRID_a_r + 1.71338624816775e-6*v_GRID_b_i + 0.00697522388628832*v_GRID_b_r - 8.56693124121928e-7*v_GRID_c_i - 0.00348761194314422*v_GRID_c_r - v_W1lv_b_r
        struct[0].g[3,0] = -0.0382049854269416*i_POI_a_i - 0.0636395371726714*i_POI_a_r + 0.0764099708538862*i_POI_b_i + 0.127279074345344*i_POI_b_r - 0.0382049854269445*i_POI_c_i - 0.0636395371726728*i_POI_c_r - 0.0280418380652299*i_POImv_a_i - 0.0971901504394967*i_POImv_a_r + 0.0280418380652406*i_POImv_b_i + 0.0971901504394895*i_POImv_b_r - 0.000634767892296777*i_W1lv_a_i - 0.00199839389307366*i_W1lv_a_r + 0.00317393578459362*i_W1lv_b_i + 0.0154231877861474*i_W1lv_b_r - 0.000634767892296846*i_W1lv_c_i - 0.00199839389307372*i_W1lv_c_r - 0.0318681229122115*i_W1mv_a_i - 0.100328108879395*i_W1mv_a_r + 0.0318681229122223*i_W1mv_b_i + 0.100328108879388*i_W1mv_b_r - 0.000609371587085066*i_W2lv_a_i - 0.00197756280128896*i_W2lv_a_r + 0.0012187431741702*i_W2lv_b_i + 0.00395512560257798*i_W2lv_b_r - 0.000609371587085134*i_W2lv_c_i - 0.00197756280128902*i_W2lv_c_r - 0.0305931173773937*i_W2mv_a_i - 0.0992822970142287*i_W2mv_a_r + 0.0305931173774048*i_W2mv_b_i + 0.0992822970142215*i_W2mv_b_r - 0.000583966813859679*i_W3lv_a_i - 0.00195672824753666*i_W3lv_a_r + 0.00116793362771943*i_W3lv_b_i + 0.00391345649507338*i_W3lv_b_r - 0.000583966813859747*i_W3lv_c_i - 0.00195672824753672*i_W3lv_c_r - 0.0293176867112742*i_W3mv_a_i - 0.098236311343156*i_W3mv_a_r + 0.029317686711285*i_W3mv_b_i + 0.0982363113431489*i_W3mv_b_r - 0.00348761194314409*v_GRID_a_i + 8.5669312404275e-7*v_GRID_a_r + 0.00697522388628832*v_GRID_b_i - 1.71338624816775e-6*v_GRID_b_r - 0.00348761194314422*v_GRID_c_i + 8.56693124121928e-7*v_GRID_c_r - v_W1lv_b_i
        struct[0].g[4,0] = 0.0636395371726713*i_POI_a_i - 0.0382049854269434*i_POI_a_r + 0.0636395371726737*i_POI_b_i - 0.0382049854269442*i_POI_b_r - 0.127279074345345*i_POI_c_i + 0.0764099708538876*i_POI_c_r + 0.0971901504394933*i_POImv_b_i - 0.0280418380652387*i_POImv_b_r - 0.0971901504394944*i_POImv_c_i + 0.0280418380652339*i_POImv_c_r + 0.00199839389307367*i_W1lv_a_i - 0.000634767892296827*i_W1lv_a_r + 0.00199839389307374*i_W1lv_b_i - 0.000634767892296831*i_W1lv_b_r - 0.0154231877861474*i_W1lv_c_i + 0.00317393578459366*i_W1lv_c_r + 0.100328108879391*i_W1mv_b_i - 0.0318681229122202*i_W1mv_b_r - 0.100328108879392*i_W1mv_c_i + 0.0318681229122156*i_W1mv_c_r + 0.00197756280128897*i_W2lv_a_i - 0.000609371587085114*i_W2lv_a_r + 0.00197756280128904*i_W2lv_b_i - 0.00060937158708512*i_W2lv_b_r - 0.00395512560257801*i_W2lv_c_i + 0.00121874317417024*i_W2lv_c_r + 0.0992822970142254*i_W2mv_b_i - 0.0305931173774027*i_W2mv_b_r - 0.0992822970142266*i_W2mv_c_i + 0.0305931173773979*i_W2mv_c_r + 0.00195672824753667*i_W3lv_a_i - 0.000583966813859728*i_W3lv_a_r + 0.00195672824753674*i_W3lv_b_i - 0.000583966813859733*i_W3lv_b_r - 0.00391345649507341*i_W3lv_c_i + 0.00116793362771946*i_W3lv_c_r + 0.0982363113431527*i_W3mv_b_i - 0.0293176867112827*i_W3mv_b_r - 0.098236311343154*i_W3mv_c_i + 0.0293176867112781*i_W3mv_c_r - 8.56693124118069e-7*v_GRID_a_i - 0.00348761194314413*v_GRID_a_r - 8.56693124090318e-7*v_GRID_b_i - 0.00348761194314425*v_GRID_b_r + 1.71338624820762e-6*v_GRID_c_i + 0.00697522388628838*v_GRID_c_r - v_W1lv_c_r
        struct[0].g[5,0] = -0.0382049854269434*i_POI_a_i - 0.0636395371726713*i_POI_a_r - 0.0382049854269442*i_POI_b_i - 0.0636395371726737*i_POI_b_r + 0.0764099708538876*i_POI_c_i + 0.127279074345345*i_POI_c_r - 0.0280418380652387*i_POImv_b_i - 0.0971901504394933*i_POImv_b_r + 0.0280418380652339*i_POImv_c_i + 0.0971901504394944*i_POImv_c_r - 0.000634767892296827*i_W1lv_a_i - 0.00199839389307367*i_W1lv_a_r - 0.000634767892296831*i_W1lv_b_i - 0.00199839389307374*i_W1lv_b_r + 0.00317393578459366*i_W1lv_c_i + 0.0154231877861474*i_W1lv_c_r - 0.0318681229122202*i_W1mv_b_i - 0.100328108879391*i_W1mv_b_r + 0.0318681229122156*i_W1mv_c_i + 0.100328108879392*i_W1mv_c_r - 0.000609371587085114*i_W2lv_a_i - 0.00197756280128897*i_W2lv_a_r - 0.00060937158708512*i_W2lv_b_i - 0.00197756280128904*i_W2lv_b_r + 0.00121874317417024*i_W2lv_c_i + 0.00395512560257801*i_W2lv_c_r - 0.0305931173774027*i_W2mv_b_i - 0.0992822970142254*i_W2mv_b_r + 0.0305931173773979*i_W2mv_c_i + 0.0992822970142266*i_W2mv_c_r - 0.000583966813859728*i_W3lv_a_i - 0.00195672824753667*i_W3lv_a_r - 0.000583966813859733*i_W3lv_b_i - 0.00195672824753674*i_W3lv_b_r + 0.00116793362771946*i_W3lv_c_i + 0.00391345649507341*i_W3lv_c_r - 0.0293176867112827*i_W3mv_b_i - 0.0982363113431527*i_W3mv_b_r + 0.0293176867112781*i_W3mv_c_i + 0.098236311343154*i_W3mv_c_r - 0.00348761194314413*v_GRID_a_i + 8.56693124118069e-7*v_GRID_a_r - 0.00348761194314425*v_GRID_b_i + 8.56693124090318e-7*v_GRID_b_r + 0.00697522388628838*v_GRID_c_i - 1.71338624820762e-6*v_GRID_c_r - v_W1lv_c_i
        struct[0].g[6,0] = -0.127279026494919*i_POI_a_i + 0.0764096462087187*i_POI_a_r + 0.0636395132474589*i_POI_b_i - 0.0382048231043588*i_POI_b_r + 0.0636395132474605*i_POI_c_i - 0.0382048231043599*i_POI_c_r - 0.0971900621093923*i_POImv_a_i + 0.0280416326518444*i_POImv_a_r + 0.0971900621093875*i_POImv_c_i - 0.0280416326518497*i_POImv_c_r - 0.00395512560257793*i_W1lv_a_i + 0.00121874317417018*i_W1lv_a_r + 0.00197756280128894*i_W1lv_b_i - 0.000609371587085081*i_W1lv_b_r + 0.00197756280128899*i_W1lv_c_i - 0.000609371587085101*i_W1lv_c_r - 0.0992822970142262*i_W1mv_a_i + 0.0305931173773962*i_W1mv_a_r + 0.0992822970142214*i_W1mv_c_i - 0.0305931173774014*i_W1mv_c_r - 0.0153815221406103*i_W2lv_a_i + 0.00312313470615651*i_W2lv_a_r + 0.00197756107030514*i_W2lv_b_i - 0.000609367353078243*i_W2lv_b_r + 0.00197756107030519*i_W2lv_c_i - 0.000609367353078263*i_W2lv_c_r - 0.099282210111273*i_W2mv_a_i + 0.030592904811745*i_W2mv_a_r + 0.0992822101112681*i_W2mv_c_i - 0.0305929048117504*i_W2mv_c_r - 0.00391345300468828*i_W3lv_a_i + 0.00116792530215105*i_W3lv_a_r + 0.00195672650234412*i_W3lv_b_i - 0.000583962651075517*i_W3lv_b_r + 0.00195672650234417*i_W3lv_c_i - 0.000583962651075537*i_W3lv_c_r - 0.0982362237268603*i_W3mv_a_i + 0.0293174777213142*i_W3mv_a_r + 0.0982362237268554*i_W3mv_c_i - 0.0293174777213195*i_W3mv_c_r + 1.70146300431068e-6*v_GRID_a_i + 0.00697521411040091*v_GRID_a_r - 8.5073150215171e-7*v_GRID_b_i - 0.00348760705520041*v_GRID_b_r - 8.50731502159103e-7*v_GRID_c_i - 0.0034876070552005*v_GRID_c_r - v_W2lv_a_r
        struct[0].g[7,0] = 0.0764096462087187*i_POI_a_i + 0.127279026494919*i_POI_a_r - 0.0382048231043588*i_POI_b_i - 0.0636395132474589*i_POI_b_r - 0.0382048231043599*i_POI_c_i - 0.0636395132474605*i_POI_c_r + 0.0280416326518444*i_POImv_a_i + 0.0971900621093923*i_POImv_a_r - 0.0280416326518497*i_POImv_c_i - 0.0971900621093875*i_POImv_c_r + 0.00121874317417018*i_W1lv_a_i + 0.00395512560257793*i_W1lv_a_r - 0.000609371587085081*i_W1lv_b_i - 0.00197756280128894*i_W1lv_b_r - 0.000609371587085101*i_W1lv_c_i - 0.00197756280128899*i_W1lv_c_r + 0.0305931173773962*i_W1mv_a_i + 0.0992822970142262*i_W1mv_a_r - 0.0305931173774014*i_W1mv_c_i - 0.0992822970142214*i_W1mv_c_r + 0.00312313470615651*i_W2lv_a_i + 0.0153815221406103*i_W2lv_a_r - 0.000609367353078243*i_W2lv_b_i - 0.00197756107030514*i_W2lv_b_r - 0.000609367353078263*i_W2lv_c_i - 0.00197756107030519*i_W2lv_c_r + 0.030592904811745*i_W2mv_a_i + 0.099282210111273*i_W2mv_a_r - 0.0305929048117504*i_W2mv_c_i - 0.0992822101112681*i_W2mv_c_r + 0.00116792530215105*i_W3lv_a_i + 0.00391345300468828*i_W3lv_a_r - 0.000583962651075517*i_W3lv_b_i - 0.00195672650234412*i_W3lv_b_r - 0.000583962651075537*i_W3lv_c_i - 0.00195672650234417*i_W3lv_c_r + 0.0293174777213142*i_W3mv_a_i + 0.0982362237268603*i_W3mv_a_r - 0.0293174777213195*i_W3mv_c_i - 0.0982362237268554*i_W3mv_c_r + 0.00697521411040091*v_GRID_a_i - 1.70146300431068e-6*v_GRID_a_r - 0.00348760705520041*v_GRID_b_i + 8.5073150215171e-7*v_GRID_b_r - 0.0034876070552005*v_GRID_c_i + 8.50731502159103e-7*v_GRID_c_r - v_W2lv_a_i
        struct[0].g[8,0] = 0.0636395132474598*i_POI_a_i - 0.0382048231043585*i_POI_a_r - 0.127279026494921*i_POI_b_i + 0.0764096462087198*i_POI_b_r + 0.0636395132474612*i_POI_c_i - 0.0382048231043613*i_POI_c_r + 0.0971900621093946*i_POImv_a_i - 0.0280416326518423*i_POImv_a_r - 0.0971900621093876*i_POImv_b_i + 0.0280416326518527*i_POImv_b_r + 0.00197756280128896*i_W1lv_a_i - 0.000609371587085065*i_W1lv_a_r - 0.00395512560257798*i_W1lv_b_i + 0.0012187431741702*i_W1lv_b_r + 0.00197756280128902*i_W1lv_c_i - 0.000609371587085133*i_W1lv_c_r + 0.0992822970142286*i_W1mv_a_i - 0.0305931173773939*i_W1mv_a_r - 0.0992822970142216*i_W1mv_b_i + 0.0305931173774045*i_W1mv_b_r + 0.00197756107030516*i_W2lv_a_i - 0.000609367353078228*i_W2lv_a_r - 0.0153815221406104*i_W2lv_b_i + 0.00312313470615652*i_W2lv_b_r + 0.00197756107030522*i_W2lv_c_i - 0.000609367353078295*i_W2lv_c_r + 0.0992822101112753*i_W2mv_a_i - 0.0305929048117428*i_W2mv_a_r - 0.0992822101112682*i_W2mv_b_i + 0.0305929048117536*i_W2mv_b_r + 0.00195672650234414*i_W3lv_a_i - 0.000583962651075503*i_W3lv_a_r - 0.00391345300468834*i_W3lv_b_i + 0.00116792530215107*i_W3lv_b_r + 0.0019567265023442*i_W3lv_c_i - 0.00058396265107557*i_W3lv_c_r + 0.0982362237268626*i_W3mv_a_i - 0.029317477721312*i_W3mv_a_r - 0.0982362237268556*i_W3mv_b_i + 0.0293174777213226*i_W3mv_b_r - 8.50731502117679e-7*v_GRID_a_i - 0.00348760705520044*v_GRID_a_r + 1.70146300431631e-6*v_GRID_b_i + 0.00697521411040101*v_GRID_b_r - 8.50731502195773e-7*v_GRID_c_i - 0.00348760705520057*v_GRID_c_r - v_W2lv_b_r
        struct[0].g[9,0] = -0.0382048231043585*i_POI_a_i - 0.0636395132474598*i_POI_a_r + 0.0764096462087198*i_POI_b_i + 0.127279026494921*i_POI_b_r - 0.0382048231043613*i_POI_c_i - 0.0636395132474612*i_POI_c_r - 0.0280416326518423*i_POImv_a_i - 0.0971900621093946*i_POImv_a_r + 0.0280416326518527*i_POImv_b_i + 0.0971900621093876*i_POImv_b_r - 0.000609371587085065*i_W1lv_a_i - 0.00197756280128896*i_W1lv_a_r + 0.0012187431741702*i_W1lv_b_i + 0.00395512560257798*i_W1lv_b_r - 0.000609371587085133*i_W1lv_c_i - 0.00197756280128902*i_W1lv_c_r - 0.0305931173773939*i_W1mv_a_i - 0.0992822970142286*i_W1mv_a_r + 0.0305931173774045*i_W1mv_b_i + 0.0992822970142216*i_W1mv_b_r - 0.000609367353078228*i_W2lv_a_i - 0.00197756107030516*i_W2lv_a_r + 0.00312313470615652*i_W2lv_b_i + 0.0153815221406104*i_W2lv_b_r - 0.000609367353078295*i_W2lv_c_i - 0.00197756107030522*i_W2lv_c_r - 0.0305929048117428*i_W2mv_a_i - 0.0992822101112753*i_W2mv_a_r + 0.0305929048117536*i_W2mv_b_i + 0.0992822101112682*i_W2mv_b_r - 0.000583962651075503*i_W3lv_a_i - 0.00195672650234414*i_W3lv_a_r + 0.00116792530215107*i_W3lv_b_i + 0.00391345300468834*i_W3lv_b_r - 0.00058396265107557*i_W3lv_c_i - 0.0019567265023442*i_W3lv_c_r - 0.029317477721312*i_W3mv_a_i - 0.0982362237268626*i_W3mv_a_r + 0.0293174777213226*i_W3mv_b_i + 0.0982362237268556*i_W3mv_b_r - 0.00348760705520044*v_GRID_a_i + 8.50731502117679e-7*v_GRID_a_r + 0.00697521411040101*v_GRID_b_i - 1.70146300431631e-6*v_GRID_b_r - 0.00348760705520057*v_GRID_c_i + 8.50731502195773e-7*v_GRID_c_r - v_W2lv_b_i
        struct[0].g[10,0] = 0.0636395132474597*i_POI_a_i - 0.0382048231043602*i_POI_a_r + 0.0636395132474621*i_POI_b_i - 0.038204823104361*i_POI_b_r - 0.127279026494922*i_POI_c_i + 0.0764096462087212*i_POI_c_r + 0.0971900621093912*i_POImv_b_i - 0.0280416326518508*i_POImv_b_r - 0.0971900621093924*i_POImv_c_i + 0.0280416326518462*i_POImv_c_r + 0.00197756280128897*i_W1lv_a_i - 0.000609371587085115*i_W1lv_a_r + 0.00197756280128904*i_W1lv_b_i - 0.000609371587085118*i_W1lv_b_r - 0.00395512560257801*i_W1lv_c_i + 0.00121874317417023*i_W1lv_c_r + 0.0992822970142252*i_W1mv_b_i - 0.0305931173774024*i_W1mv_b_r - 0.0992822970142263*i_W1mv_c_i + 0.0305931173773979*i_W1mv_c_r + 0.00197756107030517*i_W2lv_a_i - 0.000609367353078276*i_W2lv_a_r + 0.00197756107030524*i_W2lv_b_i - 0.00060936735307828*i_W2lv_b_r - 0.0153815221406104*i_W2lv_c_i + 0.00312313470615656*i_W2lv_c_r + 0.099282210111272*i_W2mv_b_i - 0.0305929048117514*i_W2mv_b_r - 0.0992822101112732*i_W2mv_c_i + 0.0305929048117468*i_W2mv_c_r + 0.00195672650234415*i_W3lv_a_i - 0.000583962651075551*i_W3lv_a_r + 0.00195672650234422*i_W3lv_b_i - 0.000583962651075555*i_W3lv_b_r - 0.00391345300468836*i_W3lv_c_i + 0.00116792530215111*i_W3lv_c_r + 0.0982362237268593*i_W3mv_b_i - 0.0293174777213204*i_W3mv_b_r - 0.0982362237268605*i_W3mv_c_i + 0.0293174777213159*i_W3mv_c_r - 8.50731502192782e-7*v_GRID_a_i - 0.00348760705520048*v_GRID_a_r - 8.50731502164163e-7*v_GRID_b_i - 0.00348760705520059*v_GRID_b_r + 1.70146300435488e-6*v_GRID_c_i + 0.00697521411040107*v_GRID_c_r - v_W2lv_c_r
        struct[0].g[11,0] = -0.0382048231043602*i_POI_a_i - 0.0636395132474597*i_POI_a_r - 0.038204823104361*i_POI_b_i - 0.0636395132474621*i_POI_b_r + 0.0764096462087212*i_POI_c_i + 0.127279026494922*i_POI_c_r - 0.0280416326518508*i_POImv_b_i - 0.0971900621093912*i_POImv_b_r + 0.0280416326518462*i_POImv_c_i + 0.0971900621093924*i_POImv_c_r - 0.000609371587085115*i_W1lv_a_i - 0.00197756280128897*i_W1lv_a_r - 0.000609371587085118*i_W1lv_b_i - 0.00197756280128904*i_W1lv_b_r + 0.00121874317417023*i_W1lv_c_i + 0.00395512560257801*i_W1lv_c_r - 0.0305931173774024*i_W1mv_b_i - 0.0992822970142252*i_W1mv_b_r + 0.0305931173773979*i_W1mv_c_i + 0.0992822970142263*i_W1mv_c_r - 0.000609367353078276*i_W2lv_a_i - 0.00197756107030517*i_W2lv_a_r - 0.00060936735307828*i_W2lv_b_i - 0.00197756107030524*i_W2lv_b_r + 0.00312313470615656*i_W2lv_c_i + 0.0153815221406104*i_W2lv_c_r - 0.0305929048117514*i_W2mv_b_i - 0.099282210111272*i_W2mv_b_r + 0.0305929048117468*i_W2mv_c_i + 0.0992822101112732*i_W2mv_c_r - 0.000583962651075551*i_W3lv_a_i - 0.00195672650234415*i_W3lv_a_r - 0.000583962651075555*i_W3lv_b_i - 0.00195672650234422*i_W3lv_b_r + 0.00116792530215111*i_W3lv_c_i + 0.00391345300468836*i_W3lv_c_r - 0.0293174777213204*i_W3mv_b_i - 0.0982362237268593*i_W3mv_b_r + 0.0293174777213159*i_W3mv_c_i + 0.0982362237268605*i_W3mv_c_r - 0.00348760705520048*v_GRID_a_i + 8.50731502192782e-7*v_GRID_a_r - 0.00348760705520059*v_GRID_b_i + 8.50731502164163e-7*v_GRID_b_r + 0.00697521411040107*v_GRID_c_i - 1.70146300435488e-6*v_GRID_c_r - v_W2lv_c_i
        struct[0].g[12,0] = -0.127278882942674*i_POI_a_i + 0.0764086722742936*i_POI_a_r + 0.0636394414713363*i_POI_b_i - 0.0382043361371463*i_POI_b_r + 0.0636394414713378*i_POI_c_i - 0.0382043361371473*i_POI_c_r - 0.0971897971186317*i_POImv_a_i + 0.0280410164125591*i_POImv_a_r + 0.0971897971186269*i_POImv_c_i - 0.0280410164125642*i_POImv_c_r - 0.00391345649507333*i_W1lv_a_i + 0.00116793362771941*i_W1lv_a_r + 0.00195672824753664*i_W1lv_b_i - 0.000583966813859694*i_W1lv_b_r + 0.00195672824753669*i_W1lv_c_i - 0.000583966813859713*i_W1lv_c_r - 0.0982363113431535*i_W1mv_a_i + 0.0293176867112763*i_W1mv_a_r + 0.0982363113431488*i_W1mv_c_i - 0.0293176867112815*i_W1mv_c_r - 0.00391345300468828*i_W2lv_a_i + 0.00116792530215105*i_W2lv_a_r + 0.00195672650234412*i_W2lv_b_i - 0.000583962651075518*i_W2lv_b_r + 0.00195672650234417*i_W2lv_c_i - 0.000583962651075537*i_W2lv_c_r - 0.0982362237268603*i_W2mv_a_i + 0.0293174777213141*i_W2mv_a_r + 0.0982362237268555*i_W2mv_c_i - 0.0293174777213194*i_W2mv_c_r - 0.0153398425335145*i_W3lv_a_i + 0.00307230032548127*i_W3lv_a_r + 0.00195672126675721*i_W3lv_b_i - 0.000583950162740626*i_W3lv_b_r + 0.00195672126675726*i_W3lv_c_i - 0.000583950162740645*i_W3lv_c_r - 0.0982359608775115*i_W3mv_a_i + 0.0293168507523132*i_W3mv_a_r + 0.0982359608775067*i_W3mv_c_i - 0.0293168507523184*i_W3mv_c_r + 1.66569333960609e-6*v_GRID_a_i + 0.00697518478272564*v_GRID_a_r - 8.32846669799631e-7*v_GRID_b_i - 0.00348759239136277*v_GRID_b_r - 8.3284666980659e-7*v_GRID_c_i - 0.00348759239136286*v_GRID_c_r - v_W3lv_a_r
        struct[0].g[13,0] = 0.0764086722742936*i_POI_a_i + 0.127278882942674*i_POI_a_r - 0.0382043361371463*i_POI_b_i - 0.0636394414713363*i_POI_b_r - 0.0382043361371473*i_POI_c_i - 0.0636394414713378*i_POI_c_r + 0.0280410164125591*i_POImv_a_i + 0.0971897971186317*i_POImv_a_r - 0.0280410164125642*i_POImv_c_i - 0.0971897971186269*i_POImv_c_r + 0.00116793362771941*i_W1lv_a_i + 0.00391345649507333*i_W1lv_a_r - 0.000583966813859694*i_W1lv_b_i - 0.00195672824753664*i_W1lv_b_r - 0.000583966813859713*i_W1lv_c_i - 0.00195672824753669*i_W1lv_c_r + 0.0293176867112763*i_W1mv_a_i + 0.0982363113431535*i_W1mv_a_r - 0.0293176867112815*i_W1mv_c_i - 0.0982363113431488*i_W1mv_c_r + 0.00116792530215105*i_W2lv_a_i + 0.00391345300468828*i_W2lv_a_r - 0.000583962651075518*i_W2lv_b_i - 0.00195672650234412*i_W2lv_b_r - 0.000583962651075537*i_W2lv_c_i - 0.00195672650234417*i_W2lv_c_r + 0.0293174777213141*i_W2mv_a_i + 0.0982362237268603*i_W2mv_a_r - 0.0293174777213194*i_W2mv_c_i - 0.0982362237268555*i_W2mv_c_r + 0.00307230032548127*i_W3lv_a_i + 0.0153398425335145*i_W3lv_a_r - 0.000583950162740626*i_W3lv_b_i - 0.00195672126675721*i_W3lv_b_r - 0.000583950162740645*i_W3lv_c_i - 0.00195672126675726*i_W3lv_c_r + 0.0293168507523132*i_W3mv_a_i + 0.0982359608775115*i_W3mv_a_r - 0.0293168507523184*i_W3mv_c_i - 0.0982359608775067*i_W3mv_c_r + 0.00697518478272564*v_GRID_a_i - 1.66569333960609e-6*v_GRID_a_r - 0.00348759239136277*v_GRID_b_i + 8.32846669799631e-7*v_GRID_b_r - 0.00348759239136286*v_GRID_c_i + 8.3284666980659e-7*v_GRID_c_r - v_W3lv_a_i
        struct[0].g[14,0] = 0.0636394414713372*i_POI_a_i - 0.0382043361371459*i_POI_a_r - 0.127278882942676*i_POI_b_i + 0.0764086722742947*i_POI_b_r + 0.0636394414713386*i_POI_c_i - 0.0382043361371487*i_POI_c_r + 0.097189797118634*i_POImv_a_i - 0.028041016412557*i_POImv_a_r - 0.0971897971186271*i_POImv_b_i + 0.0280410164125671*i_POImv_b_r + 0.00195672824753666*i_W1lv_a_i - 0.00058396681385968*i_W1lv_a_r - 0.00391345649507338*i_W1lv_b_i + 0.00116793362771943*i_W1lv_b_r + 0.00195672824753672*i_W1lv_c_i - 0.000583966813859745*i_W1lv_c_r + 0.0982363113431559*i_W1mv_a_i - 0.0293176867112742*i_W1mv_a_r - 0.098236311343149*i_W1mv_b_i + 0.0293176867112845*i_W1mv_b_r + 0.00195672650234414*i_W2lv_a_i - 0.000583962651075503*i_W2lv_a_r - 0.00391345300468834*i_W2lv_b_i + 0.00116792530215107*i_W2lv_b_r + 0.0019567265023442*i_W2lv_c_i - 0.000583962651075569*i_W2lv_c_r + 0.0982362237268626*i_W2mv_a_i - 0.029317477721312*i_W2mv_a_r - 0.0982362237268557*i_W2mv_b_i + 0.0293174777213224*i_W2mv_b_r + 0.00195672126675723*i_W3lv_a_i - 0.000583950162740612*i_W3lv_a_r - 0.0153398425335145*i_W3lv_b_i + 0.00307230032548129*i_W3lv_b_r + 0.00195672126675729*i_W3lv_c_i - 0.000583950162740677*i_W3lv_c_r + 0.0982359608775139*i_W3mv_a_i - 0.0293168507523111*i_W3mv_a_r - 0.0982359608775069*i_W3mv_b_i + 0.0293168507523214*i_W3mv_b_r - 8.32846669765817e-7*v_GRID_a_i - 0.0034875923913628*v_GRID_a_r + 1.66569333961041e-6*v_GRID_b_i + 0.00697518478272573*v_GRID_b_r - 8.32846669843044e-7*v_GRID_c_i - 0.00348759239136293*v_GRID_c_r - v_W3lv_b_r
        struct[0].g[15,0] = -0.0382043361371459*i_POI_a_i - 0.0636394414713372*i_POI_a_r + 0.0764086722742947*i_POI_b_i + 0.127278882942676*i_POI_b_r - 0.0382043361371487*i_POI_c_i - 0.0636394414713386*i_POI_c_r - 0.028041016412557*i_POImv_a_i - 0.097189797118634*i_POImv_a_r + 0.0280410164125671*i_POImv_b_i + 0.0971897971186271*i_POImv_b_r - 0.00058396681385968*i_W1lv_a_i - 0.00195672824753666*i_W1lv_a_r + 0.00116793362771943*i_W1lv_b_i + 0.00391345649507338*i_W1lv_b_r - 0.000583966813859745*i_W1lv_c_i - 0.00195672824753672*i_W1lv_c_r - 0.0293176867112742*i_W1mv_a_i - 0.0982363113431559*i_W1mv_a_r + 0.0293176867112845*i_W1mv_b_i + 0.098236311343149*i_W1mv_b_r - 0.000583962651075503*i_W2lv_a_i - 0.00195672650234414*i_W2lv_a_r + 0.00116792530215107*i_W2lv_b_i + 0.00391345300468834*i_W2lv_b_r - 0.000583962651075569*i_W2lv_c_i - 0.0019567265023442*i_W2lv_c_r - 0.029317477721312*i_W2mv_a_i - 0.0982362237268626*i_W2mv_a_r + 0.0293174777213224*i_W2mv_b_i + 0.0982362237268557*i_W2mv_b_r - 0.000583950162740612*i_W3lv_a_i - 0.00195672126675723*i_W3lv_a_r + 0.00307230032548129*i_W3lv_b_i + 0.0153398425335145*i_W3lv_b_r - 0.000583950162740677*i_W3lv_c_i - 0.00195672126675729*i_W3lv_c_r - 0.0293168507523111*i_W3mv_a_i - 0.0982359608775139*i_W3mv_a_r + 0.0293168507523214*i_W3mv_b_i + 0.0982359608775069*i_W3mv_b_r - 0.0034875923913628*v_GRID_a_i + 8.32846669765817e-7*v_GRID_a_r + 0.00697518478272573*v_GRID_b_i - 1.66569333961041e-6*v_GRID_b_r - 0.00348759239136293*v_GRID_c_i + 8.32846669843044e-7*v_GRID_c_r - v_W3lv_b_i
        struct[0].g[16,0] = 0.063639441471337*i_POI_a_i - 0.0382043361371477*i_POI_a_r + 0.0636394414713394*i_POI_b_i - 0.0382043361371484*i_POI_b_r - 0.127278882942676*i_POI_c_i + 0.0764086722742961*i_POI_c_r + 0.0971897971186306*i_POImv_b_i - 0.0280410164125652*i_POImv_b_r - 0.0971897971186319*i_POImv_c_i + 0.0280410164125607*i_POImv_c_r + 0.00195672824753667*i_W1lv_a_i - 0.000583966813859728*i_W1lv_a_r + 0.00195672824753674*i_W1lv_b_i - 0.000583966813859731*i_W1lv_b_r - 0.00391345649507341*i_W1lv_c_i + 0.00116793362771946*i_W1lv_c_r + 0.0982363113431526*i_W1mv_b_i - 0.0293176867112824*i_W1mv_b_r - 0.0982363113431537*i_W1mv_c_i + 0.0293176867112781*i_W1mv_c_r + 0.00195672650234415*i_W2lv_a_i - 0.000583962651075551*i_W2lv_a_r + 0.00195672650234422*i_W2lv_b_i - 0.000583962651075554*i_W2lv_b_r - 0.00391345300468836*i_W2lv_c_i + 0.00116792530215111*i_W2lv_c_r + 0.0982362237268593*i_W2mv_b_i - 0.0293174777213203*i_W2mv_b_r - 0.0982362237268605*i_W2mv_c_i + 0.0293174777213159*i_W2mv_c_r + 0.00195672126675724*i_W3lv_a_i - 0.000583950162740659*i_W3lv_a_r + 0.00195672126675731*i_W3lv_b_i - 0.000583950162740663*i_W3lv_b_r - 0.0153398425335145*i_W3lv_c_i + 0.00307230032548132*i_W3lv_c_r + 0.0982359608775105*i_W3mv_b_i - 0.0293168507523193*i_W3mv_b_r - 0.0982359608775118*i_W3mv_c_i + 0.0293168507523149*i_W3mv_c_r - 8.32846669840052e-7*v_GRID_a_i - 0.00348759239136284*v_GRID_a_r - 8.32846669810349e-7*v_GRID_b_i - 0.00348759239136295*v_GRID_b_r + 1.66569333964898e-6*v_GRID_c_i + 0.00697518478272579*v_GRID_c_r - v_W3lv_c_r
        struct[0].g[17,0] = -0.0382043361371477*i_POI_a_i - 0.063639441471337*i_POI_a_r - 0.0382043361371484*i_POI_b_i - 0.0636394414713394*i_POI_b_r + 0.0764086722742961*i_POI_c_i + 0.127278882942676*i_POI_c_r - 0.0280410164125652*i_POImv_b_i - 0.0971897971186306*i_POImv_b_r + 0.0280410164125607*i_POImv_c_i + 0.0971897971186319*i_POImv_c_r - 0.000583966813859728*i_W1lv_a_i - 0.00195672824753667*i_W1lv_a_r - 0.000583966813859731*i_W1lv_b_i - 0.00195672824753674*i_W1lv_b_r + 0.00116793362771946*i_W1lv_c_i + 0.00391345649507341*i_W1lv_c_r - 0.0293176867112824*i_W1mv_b_i - 0.0982363113431526*i_W1mv_b_r + 0.0293176867112781*i_W1mv_c_i + 0.0982363113431537*i_W1mv_c_r - 0.000583962651075551*i_W2lv_a_i - 0.00195672650234415*i_W2lv_a_r - 0.000583962651075554*i_W2lv_b_i - 0.00195672650234422*i_W2lv_b_r + 0.00116792530215111*i_W2lv_c_i + 0.00391345300468836*i_W2lv_c_r - 0.0293174777213203*i_W2mv_b_i - 0.0982362237268593*i_W2mv_b_r + 0.0293174777213159*i_W2mv_c_i + 0.0982362237268605*i_W2mv_c_r - 0.000583950162740659*i_W3lv_a_i - 0.00195672126675724*i_W3lv_a_r - 0.000583950162740663*i_W3lv_b_i - 0.00195672126675731*i_W3lv_b_r + 0.00307230032548132*i_W3lv_c_i + 0.0153398425335145*i_W3lv_c_r - 0.0293168507523193*i_W3mv_b_i - 0.0982359608775105*i_W3mv_b_r + 0.0293168507523149*i_W3mv_c_i + 0.0982359608775118*i_W3mv_c_r - 0.00348759239136284*v_GRID_a_i + 8.32846669840052e-7*v_GRID_a_r - 0.00348759239136295*v_GRID_b_i + 8.32846669810349e-7*v_GRID_b_r + 0.00697518478272579*v_GRID_c_i - 1.66569333964898e-6*v_GRID_c_r - v_W3lv_c_i
        struct[0].g[18,0] = -3.19497213887022*i_POI_a_i + 1.9179839277919*i_POI_a_r + 3.19497213887041*i_POI_b_i - 1.9179839277919*i_POI_b_r - 3.08885814101954*i_POImv_a_i + 11.6022742434667*i_POImv_a_r + 1.79047234076949*i_POImv_b_i + 10.1945442087447*i_POImv_b_r + 1.79047234076923*i_POImv_c_i + 10.1945442087447*i_POImv_c_r - 0.0971901504394881*i_W1lv_a_i + 0.028041838065235*i_W1lv_a_r + 0.0971901504394933*i_W1lv_b_i - 0.0280418380652335*i_W1lv_b_r - 3.08738963687028*i_W1mv_a_i + 11.6035242966407*i_W1mv_a_r + 1.79198075607073*i_W1mv_b_i + 10.1957014483333*i_W1mv_b_r + 1.79198075607047*i_W1mv_c_i + 10.1957014483332*i_W1mv_c_r - 0.0971900621093861*i_W2lv_a_i + 0.0280416326518473*i_W2lv_a_r + 0.0971900621093912*i_W2lv_b_i - 0.0280416326518457*i_W2lv_b_r - 3.08755280951041*i_W2mv_a_i + 11.60338540301*i_W2mv_a_r + 1.79181314887337*i_W2mv_b_i + 10.1955728673526*i_W2mv_b_r + 1.79181314887311*i_W2mv_c_i + 10.1955728673525*i_W2mv_c_r - 0.0971897971186256*i_W3lv_a_i + 0.028041016412562*i_W3lv_a_r + 0.0971897971186307*i_W3lv_b_i - 0.0280410164125604*i_W3lv_b_r - 3.08804231916211*i_W3mv_a_i + 11.6029687203683*i_W3mv_a_r + 1.79131033552716*i_W3mv_b_i + 10.1951871226168*i_W3mv_b_r + 1.7913103355269*i_W3mv_c_i + 10.1951871226167*i_W3mv_c_r + 4.03160543824476e-5*v_GRID_a_i + 0.175091156146066*v_GRID_a_r - 4.03160543774783e-5*v_GRID_b_i - 0.175091156146074*v_GRID_b_r - 3.9510886540828e-17*v_GRID_c_i + 3.80491223180095e-17*v_GRID_c_r - v_POImv_a_r
        struct[0].g[19,0] = 1.9179839277919*i_POI_a_i + 3.19497213887022*i_POI_a_r - 1.9179839277919*i_POI_b_i - 3.19497213887041*i_POI_b_r + 11.6022742434667*i_POImv_a_i + 3.08885814101954*i_POImv_a_r + 10.1945442087447*i_POImv_b_i - 1.79047234076949*i_POImv_b_r + 10.1945442087447*i_POImv_c_i - 1.79047234076923*i_POImv_c_r + 0.028041838065235*i_W1lv_a_i + 0.0971901504394881*i_W1lv_a_r - 0.0280418380652335*i_W1lv_b_i - 0.0971901504394933*i_W1lv_b_r + 11.6035242966407*i_W1mv_a_i + 3.08738963687028*i_W1mv_a_r + 10.1957014483333*i_W1mv_b_i - 1.79198075607073*i_W1mv_b_r + 10.1957014483332*i_W1mv_c_i - 1.79198075607047*i_W1mv_c_r + 0.0280416326518473*i_W2lv_a_i + 0.0971900621093861*i_W2lv_a_r - 0.0280416326518457*i_W2lv_b_i - 0.0971900621093912*i_W2lv_b_r + 11.60338540301*i_W2mv_a_i + 3.08755280951041*i_W2mv_a_r + 10.1955728673526*i_W2mv_b_i - 1.79181314887337*i_W2mv_b_r + 10.1955728673525*i_W2mv_c_i - 1.79181314887311*i_W2mv_c_r + 0.028041016412562*i_W3lv_a_i + 0.0971897971186256*i_W3lv_a_r - 0.0280410164125604*i_W3lv_b_i - 0.0971897971186307*i_W3lv_b_r + 11.6029687203683*i_W3mv_a_i + 3.08804231916211*i_W3mv_a_r + 10.1951871226168*i_W3mv_b_i - 1.79131033552716*i_W3mv_b_r + 10.1951871226167*i_W3mv_c_i - 1.7913103355269*i_W3mv_c_r + 0.175091156146066*v_GRID_a_i - 4.03160543824476e-5*v_GRID_a_r - 0.175091156146074*v_GRID_b_i + 4.03160543774783e-5*v_GRID_b_r + 3.80491223180095e-17*v_GRID_c_i + 3.9510886540828e-17*v_GRID_c_r - v_POImv_a_i
        struct[0].g[20,0] = -3.19497213887037*i_POI_b_i + 1.91798392779201*i_POI_b_r + 3.19497213887023*i_POI_c_i - 1.91798392779203*i_POI_c_r + 1.79047234076965*i_POImv_a_i + 10.1945442087449*i_POImv_a_r - 3.08885814101935*i_POImv_b_i + 11.6022742434671*i_POImv_b_r + 1.79047234076948*i_POImv_c_i + 10.1945442087448*i_POImv_c_r - 0.097190150439493*i_W1lv_b_i + 0.028041838065237*i_W1lv_b_r + 0.0971901504394895*i_W1lv_c_i - 0.0280418380652384*i_W1lv_c_r + 1.7919807560709*i_W1mv_a_i + 10.1957014483335*i_W1mv_a_r - 3.0873896368701*i_W1mv_b_i + 11.603524296641*i_W1mv_b_r + 1.79198075607072*i_W1mv_c_i + 10.1957014483334*i_W1mv_c_r - 0.097190062109391*i_W2lv_b_i + 0.0280416326518492*i_W2lv_b_r + 0.0971900621093875*i_W2lv_c_i - 0.0280416326518506*i_W2lv_c_r + 1.79181314887354*i_W2mv_a_i + 10.1955728673528*i_W2mv_a_r - 3.08755280951023*i_W2mv_b_i + 11.6033854030103*i_W2mv_b_r + 1.79181314887336*i_W2mv_c_i + 10.1955728673527*i_W2mv_c_r - 0.0971897971186304*i_W3lv_b_i + 0.0280410164125637*i_W3lv_b_r + 0.0971897971186269*i_W3lv_c_i - 0.0280410164125652*i_W3lv_c_r + 1.79131033552732*i_W3mv_a_i + 10.1951871226169*i_W3mv_a_r - 3.08804231916193*i_W3mv_b_i + 11.6029687203686*i_W3mv_b_r + 1.79131033552715*i_W3mv_c_i + 10.1951871226169*i_W3mv_c_r + 4.89567608905926e-18*v_GRID_a_i + 2.93587278316694e-18*v_GRID_a_r + 4.03160543830677e-5*v_GRID_b_i + 0.175091156146075*v_GRID_b_r - 4.03160543870874e-5*v_GRID_c_i - 0.17509115614607*v_GRID_c_r - v_POImv_b_r
        struct[0].g[21,0] = 1.91798392779201*i_POI_b_i + 3.19497213887037*i_POI_b_r - 1.91798392779203*i_POI_c_i - 3.19497213887023*i_POI_c_r + 10.1945442087449*i_POImv_a_i - 1.79047234076965*i_POImv_a_r + 11.6022742434671*i_POImv_b_i + 3.08885814101935*i_POImv_b_r + 10.1945442087448*i_POImv_c_i - 1.79047234076948*i_POImv_c_r + 0.028041838065237*i_W1lv_b_i + 0.097190150439493*i_W1lv_b_r - 0.0280418380652384*i_W1lv_c_i - 0.0971901504394895*i_W1lv_c_r + 10.1957014483335*i_W1mv_a_i - 1.7919807560709*i_W1mv_a_r + 11.603524296641*i_W1mv_b_i + 3.0873896368701*i_W1mv_b_r + 10.1957014483334*i_W1mv_c_i - 1.79198075607072*i_W1mv_c_r + 0.0280416326518492*i_W2lv_b_i + 0.097190062109391*i_W2lv_b_r - 0.0280416326518506*i_W2lv_c_i - 0.0971900621093875*i_W2lv_c_r + 10.1955728673528*i_W2mv_a_i - 1.79181314887354*i_W2mv_a_r + 11.6033854030103*i_W2mv_b_i + 3.08755280951023*i_W2mv_b_r + 10.1955728673527*i_W2mv_c_i - 1.79181314887336*i_W2mv_c_r + 0.0280410164125637*i_W3lv_b_i + 0.0971897971186304*i_W3lv_b_r - 0.0280410164125652*i_W3lv_c_i - 0.0971897971186269*i_W3lv_c_r + 10.1951871226169*i_W3mv_a_i - 1.79131033552732*i_W3mv_a_r + 11.6029687203686*i_W3mv_b_i + 3.08804231916193*i_W3mv_b_r + 10.1951871226169*i_W3mv_c_i - 1.79131033552715*i_W3mv_c_r + 2.93587278316694e-18*v_GRID_a_i - 4.89567608905926e-18*v_GRID_a_r + 0.175091156146075*v_GRID_b_i - 4.03160543830677e-5*v_GRID_b_r - 0.17509115614607*v_GRID_c_i + 4.03160543870874e-5*v_GRID_c_r - v_POImv_b_i
        struct[0].g[22,0] = 3.19497213887049*i_POI_a_i - 1.91798392779195*i_POI_a_r - 3.19497213887058*i_POI_c_i + 1.91798392779195*i_POI_c_r + 1.79047234076953*i_POImv_a_i + 10.1945442087448*i_POImv_a_r + 1.79047234076966*i_POImv_b_i + 10.1945442087448*i_POImv_b_r - 3.0888581410196*i_POImv_c_i + 11.6022742434669*i_POImv_c_r + 0.0971901504394956*i_W1lv_a_i - 0.0280418380652345*i_W1lv_a_r - 0.0971901504394982*i_W1lv_c_i + 0.0280418380652337*i_W1lv_c_r + 1.79198075607078*i_W1mv_a_i + 10.1957014483334*i_W1mv_a_r + 1.79198075607091*i_W1mv_b_i + 10.1957014483334*i_W1mv_b_r - 3.08738963687035*i_W1mv_c_i + 11.6035242966408*i_W1mv_c_r + 0.0971900621093936*i_W2lv_a_i - 0.0280416326518468*i_W2lv_a_r - 0.0971900621093962*i_W2lv_c_i + 0.028041632651846*i_W2lv_c_r + 1.79181314887342*i_W2mv_a_i + 10.1955728673527*i_W2mv_a_r + 1.79181314887355*i_W2mv_b_i + 10.1955728673527*i_W2mv_b_r - 3.08755280951048*i_W2mv_c_i + 11.6033854030101*i_W2mv_c_r + 0.0971897971186331*i_W3lv_a_i - 0.0280410164125613*i_W3lv_a_r - 0.0971897971186357*i_W3lv_c_i + 0.0280410164125606*i_W3lv_c_r + 1.79131033552721*i_W3mv_a_i + 10.1951871226168*i_W3mv_a_r + 1.79131033552734*i_W3mv_b_i + 10.1951871226169*i_W3mv_b_r - 3.08804231916218*i_W3mv_c_i + 11.6029687203684*i_W3mv_c_r - 4.03160543779422e-5*v_GRID_a_i - 0.175091156146078*v_GRID_a_r + 3.10999136728748e-17*v_GRID_b_i - 4.08928123428111e-17*v_GRID_b_r + 4.03160543753863e-5*v_GRID_c_i + 0.175091156146082*v_GRID_c_r - v_POImv_c_r
        struct[0].g[23,0] = -1.91798392779195*i_POI_a_i - 3.19497213887049*i_POI_a_r + 1.91798392779195*i_POI_c_i + 3.19497213887058*i_POI_c_r + 10.1945442087448*i_POImv_a_i - 1.79047234076953*i_POImv_a_r + 10.1945442087448*i_POImv_b_i - 1.79047234076966*i_POImv_b_r + 11.6022742434669*i_POImv_c_i + 3.0888581410196*i_POImv_c_r - 0.0280418380652345*i_W1lv_a_i - 0.0971901504394956*i_W1lv_a_r + 0.0280418380652337*i_W1lv_c_i + 0.0971901504394982*i_W1lv_c_r + 10.1957014483334*i_W1mv_a_i - 1.79198075607078*i_W1mv_a_r + 10.1957014483334*i_W1mv_b_i - 1.79198075607091*i_W1mv_b_r + 11.6035242966408*i_W1mv_c_i + 3.08738963687035*i_W1mv_c_r - 0.0280416326518468*i_W2lv_a_i - 0.0971900621093936*i_W2lv_a_r + 0.028041632651846*i_W2lv_c_i + 0.0971900621093962*i_W2lv_c_r + 10.1955728673527*i_W2mv_a_i - 1.79181314887342*i_W2mv_a_r + 10.1955728673527*i_W2mv_b_i - 1.79181314887355*i_W2mv_b_r + 11.6033854030101*i_W2mv_c_i + 3.08755280951048*i_W2mv_c_r - 0.0280410164125613*i_W3lv_a_i - 0.0971897971186331*i_W3lv_a_r + 0.0280410164125606*i_W3lv_c_i + 0.0971897971186357*i_W3lv_c_r + 10.1951871226168*i_W3mv_a_i - 1.79131033552721*i_W3mv_a_r + 10.1951871226169*i_W3mv_b_i - 1.79131033552734*i_W3mv_b_r + 11.6029687203684*i_W3mv_c_i + 3.08804231916218*i_W3mv_c_r - 0.175091156146078*v_GRID_a_i + 4.03160543779422e-5*v_GRID_a_r - 4.08928123428111e-17*v_GRID_b_i - 3.10999136728748e-17*v_GRID_b_r + 0.175091156146082*v_GRID_c_i - 4.03160543753863e-5*v_GRID_c_r - v_POImv_c_i
        struct[0].g[24,0] = -16.3488065237228*i_POI_a_i + 8.99352976113163*i_POI_a_r + 1.90429395893588*i_POI_b_i - 1.96237550602395*i_POI_b_r + 1.90429395893593*i_POI_c_i - 1.962375506024*i_POI_c_r - 3.19497213887046*i_POImv_a_i + 1.91798392779186*i_POImv_a_r + 3.19497213887024*i_POImv_c_i - 1.91798392779199*i_POImv_c_r - 0.127279074345343*i_W1lv_a_i + 0.076409970853885*i_W1lv_a_r + 0.0636395371726706*i_W1lv_b_i - 0.0382049854269419*i_W1lv_b_r + 0.0636395371726721*i_W1lv_c_i - 0.038204985426943*i_W1lv_c_r - 3.19498294936924*i_W1mv_a_i + 1.91805727135914*i_W1mv_a_r + 3.19498294936902*i_W1mv_c_i - 1.91805727135928*i_W1mv_c_r - 0.127279026494919*i_W2lv_a_i + 0.0764096462087186*i_W2lv_a_r + 0.063639513247459*i_W2lv_b_i - 0.0382048231043588*i_W2lv_b_r + 0.0636395132474605*i_W2lv_c_i - 0.0382048231043599*i_W2lv_c_r - 3.19498174821903*i_W2mv_a_i + 1.91804912205592*i_W2mv_a_r + 3.19498174821881*i_W2mv_c_i - 1.91804912205606*i_W2mv_c_r - 0.127278882942674*i_W3lv_a_i + 0.0764086722742935*i_W3lv_a_r + 0.0636394414713363*i_W3lv_b_i - 0.0382043361371462*i_W3lv_b_r + 0.0636394414713379*i_W3lv_c_i - 0.0382043361371473*i_W3lv_c_r - 3.19497814474392*i_W3mv_a_i + 1.9180246741732*i_W3mv_a_r + 3.19497814474371*i_W3mv_c_i - 1.91802467417334*i_W3mv_c_r - 0.0328668071354569*v_GRID_a_i + 0.876104930717234*v_GRID_a_r - 0.0330297796399041*v_GRID_b_i - 0.124162742246183*v_GRID_b_r - 0.0330297796399051*v_GRID_c_i - 0.124162742246186*v_GRID_c_r - v_POI_a_r
        struct[0].g[25,0] = 8.99352976113163*i_POI_a_i + 16.3488065237228*i_POI_a_r - 1.96237550602395*i_POI_b_i - 1.90429395893588*i_POI_b_r - 1.962375506024*i_POI_c_i - 1.90429395893593*i_POI_c_r + 1.91798392779186*i_POImv_a_i + 3.19497213887046*i_POImv_a_r - 1.91798392779199*i_POImv_c_i - 3.19497213887024*i_POImv_c_r + 0.076409970853885*i_W1lv_a_i + 0.127279074345343*i_W1lv_a_r - 0.0382049854269419*i_W1lv_b_i - 0.0636395371726706*i_W1lv_b_r - 0.038204985426943*i_W1lv_c_i - 0.0636395371726721*i_W1lv_c_r + 1.91805727135914*i_W1mv_a_i + 3.19498294936924*i_W1mv_a_r - 1.91805727135928*i_W1mv_c_i - 3.19498294936902*i_W1mv_c_r + 0.0764096462087186*i_W2lv_a_i + 0.127279026494919*i_W2lv_a_r - 0.0382048231043588*i_W2lv_b_i - 0.063639513247459*i_W2lv_b_r - 0.0382048231043599*i_W2lv_c_i - 0.0636395132474605*i_W2lv_c_r + 1.91804912205592*i_W2mv_a_i + 3.19498174821903*i_W2mv_a_r - 1.91804912205606*i_W2mv_c_i - 3.19498174821881*i_W2mv_c_r + 0.0764086722742935*i_W3lv_a_i + 0.127278882942674*i_W3lv_a_r - 0.0382043361371462*i_W3lv_b_i - 0.0636394414713363*i_W3lv_b_r - 0.0382043361371473*i_W3lv_c_i - 0.0636394414713379*i_W3lv_c_r + 1.9180246741732*i_W3mv_a_i + 3.19497814474392*i_W3mv_a_r - 1.91802467417334*i_W3mv_c_i - 3.19497814474371*i_W3mv_c_r + 0.876104930717234*v_GRID_a_i + 0.0328668071354569*v_GRID_a_r - 0.124162742246183*v_GRID_b_i + 0.0330297796399041*v_GRID_b_r - 0.124162742246186*v_GRID_c_i + 0.0330297796399051*v_GRID_c_r - v_POI_a_i
        struct[0].g[26,0] = 1.90429395893591*i_POI_a_i - 1.96237550602395*i_POI_a_r - 16.3488065237228*i_POI_b_i + 8.99352976113168*i_POI_b_r + 1.90429395893593*i_POI_c_i - 1.96237550602406*i_POI_c_r + 3.19497213887056*i_POImv_a_i - 1.91798392779181*i_POImv_a_r - 3.19497213887022*i_POImv_b_i + 1.9179839277921*i_POImv_b_r + 0.0636395371726714*i_W1lv_a_i - 0.0382049854269416*i_W1lv_a_r - 0.127279074345344*i_W1lv_b_i + 0.0764099708538861*i_W1lv_b_r + 0.0636395371726729*i_W1lv_c_i - 0.0382049854269445*i_W1lv_c_r + 3.19498294936934*i_W1mv_a_i - 1.91805727135909*i_W1mv_a_r - 3.194982949369*i_W1mv_b_i + 1.91805727135939*i_W1mv_b_r + 0.0636395132474598*i_W2lv_a_i - 0.0382048231043584*i_W2lv_a_r - 0.127279026494921*i_W2lv_b_i + 0.0764096462087197*i_W2lv_b_r + 0.0636395132474612*i_W2lv_c_i - 0.0382048231043613*i_W2lv_c_r + 3.19498174821913*i_W2mv_a_i - 1.91804912205587*i_W2mv_a_r - 3.19498174821879*i_W2mv_b_i + 1.91804912205617*i_W2mv_b_r + 0.0636394414713371*i_W3lv_a_i - 0.0382043361371459*i_W3lv_a_r - 0.127278882942676*i_W3lv_b_i + 0.0764086722742946*i_W3lv_b_r + 0.0636394414713386*i_W3lv_c_i - 0.0382043361371488*i_W3lv_c_r + 3.19497814474403*i_W3mv_a_i - 1.91802467417315*i_W3mv_a_r - 3.19497814474369*i_W3mv_b_i + 1.91802467417345*i_W3mv_b_r - 0.0330297796399032*v_GRID_a_i - 0.124162742246184*v_GRID_a_r - 0.0328668071354559*v_GRID_b_i + 0.876104930717237*v_GRID_b_r - 0.033029779639907*v_GRID_c_i - 0.124162742246188*v_GRID_c_r - v_POI_b_r
        struct[0].g[27,0] = -1.96237550602395*i_POI_a_i - 1.90429395893591*i_POI_a_r + 8.99352976113168*i_POI_b_i + 16.3488065237228*i_POI_b_r - 1.96237550602406*i_POI_c_i - 1.90429395893593*i_POI_c_r - 1.91798392779181*i_POImv_a_i - 3.19497213887056*i_POImv_a_r + 1.9179839277921*i_POImv_b_i + 3.19497213887022*i_POImv_b_r - 0.0382049854269416*i_W1lv_a_i - 0.0636395371726714*i_W1lv_a_r + 0.0764099708538861*i_W1lv_b_i + 0.127279074345344*i_W1lv_b_r - 0.0382049854269445*i_W1lv_c_i - 0.0636395371726729*i_W1lv_c_r - 1.91805727135909*i_W1mv_a_i - 3.19498294936934*i_W1mv_a_r + 1.91805727135939*i_W1mv_b_i + 3.194982949369*i_W1mv_b_r - 0.0382048231043584*i_W2lv_a_i - 0.0636395132474598*i_W2lv_a_r + 0.0764096462087197*i_W2lv_b_i + 0.127279026494921*i_W2lv_b_r - 0.0382048231043613*i_W2lv_c_i - 0.0636395132474612*i_W2lv_c_r - 1.91804912205587*i_W2mv_a_i - 3.19498174821913*i_W2mv_a_r + 1.91804912205617*i_W2mv_b_i + 3.19498174821879*i_W2mv_b_r - 0.0382043361371459*i_W3lv_a_i - 0.0636394414713371*i_W3lv_a_r + 0.0764086722742946*i_W3lv_b_i + 0.127278882942676*i_W3lv_b_r - 0.0382043361371488*i_W3lv_c_i - 0.0636394414713386*i_W3lv_c_r - 1.91802467417315*i_W3mv_a_i - 3.19497814474403*i_W3mv_a_r + 1.91802467417345*i_W3mv_b_i + 3.19497814474369*i_W3mv_b_r - 0.124162742246184*v_GRID_a_i + 0.0330297796399032*v_GRID_a_r + 0.876104930717237*v_GRID_b_i + 0.0328668071354559*v_GRID_b_r - 0.124162742246188*v_GRID_c_i + 0.033029779639907*v_GRID_c_r - v_POI_b_i
        struct[0].g[28,0] = 1.90429395893589*i_POI_a_i - 1.96237550602401*i_POI_a_r + 1.90429395893597*i_POI_b_i - 1.96237550602406*i_POI_b_r - 16.3488065237228*i_POI_c_i + 8.99352976113174*i_POI_c_r + 3.19497213887036*i_POImv_b_i - 1.91798392779206*i_POImv_b_r - 3.19497213887045*i_POImv_c_i + 1.91798392779192*i_POImv_c_r + 0.0636395371726713*i_W1lv_a_i - 0.0382049854269434*i_W1lv_a_r + 0.0636395371726737*i_W1lv_b_i - 0.0382049854269442*i_W1lv_b_r - 0.127279074345345*i_W1lv_c_i + 0.0764099708538875*i_W1lv_c_r + 3.19498294936915*i_W1mv_b_i - 1.91805727135935*i_W1mv_b_r - 3.19498294936923*i_W1mv_c_i + 1.91805727135921*i_W1mv_c_r + 0.0636395132474596*i_W2lv_a_i - 0.0382048231043602*i_W2lv_a_r + 0.0636395132474621*i_W2lv_b_i - 0.038204823104361*i_W2lv_b_r - 0.127279026494922*i_W2lv_c_i + 0.0764096462087212*i_W2lv_c_r + 3.19498174821894*i_W2mv_b_i - 1.91804912205613*i_W2mv_b_r - 3.19498174821902*i_W2mv_c_i + 1.91804912205598*i_W2mv_c_r + 0.063639441471337*i_W3lv_a_i - 0.0382043361371476*i_W3lv_a_r + 0.0636394414713394*i_W3lv_b_i - 0.0382043361371484*i_W3lv_b_r - 0.127278882942676*i_W3lv_c_i + 0.0764086722742961*i_W3lv_c_r + 3.19497814474383*i_W3mv_b_i - 1.91802467417341*i_W3mv_b_r - 3.19497814474392*i_W3mv_c_i + 1.91802467417326*i_W3mv_c_r - 0.0330297796399061*v_GRID_a_i - 0.124162742246184*v_GRID_a_r - 0.0330297796399062*v_GRID_b_i - 0.124162742246189*v_GRID_b_r - 0.032866807135454*v_GRID_c_i + 0.876104930717239*v_GRID_c_r - v_POI_c_r
        struct[0].g[29,0] = -1.96237550602401*i_POI_a_i - 1.90429395893589*i_POI_a_r - 1.96237550602406*i_POI_b_i - 1.90429395893597*i_POI_b_r + 8.99352976113174*i_POI_c_i + 16.3488065237228*i_POI_c_r - 1.91798392779206*i_POImv_b_i - 3.19497213887036*i_POImv_b_r + 1.91798392779192*i_POImv_c_i + 3.19497213887045*i_POImv_c_r - 0.0382049854269434*i_W1lv_a_i - 0.0636395371726713*i_W1lv_a_r - 0.0382049854269442*i_W1lv_b_i - 0.0636395371726737*i_W1lv_b_r + 0.0764099708538875*i_W1lv_c_i + 0.127279074345345*i_W1lv_c_r - 1.91805727135935*i_W1mv_b_i - 3.19498294936915*i_W1mv_b_r + 1.91805727135921*i_W1mv_c_i + 3.19498294936923*i_W1mv_c_r - 0.0382048231043602*i_W2lv_a_i - 0.0636395132474596*i_W2lv_a_r - 0.038204823104361*i_W2lv_b_i - 0.0636395132474621*i_W2lv_b_r + 0.0764096462087212*i_W2lv_c_i + 0.127279026494922*i_W2lv_c_r - 1.91804912205613*i_W2mv_b_i - 3.19498174821894*i_W2mv_b_r + 1.91804912205598*i_W2mv_c_i + 3.19498174821902*i_W2mv_c_r - 0.0382043361371476*i_W3lv_a_i - 0.063639441471337*i_W3lv_a_r - 0.0382043361371484*i_W3lv_b_i - 0.0636394414713394*i_W3lv_b_r + 0.0764086722742961*i_W3lv_c_i + 0.127278882942676*i_W3lv_c_r - 1.91802467417341*i_W3mv_b_i - 3.19497814474383*i_W3mv_b_r + 1.91802467417326*i_W3mv_c_i + 3.19497814474392*i_W3mv_c_r - 0.124162742246184*v_GRID_a_i + 0.0330297796399061*v_GRID_a_r - 0.124162742246189*v_GRID_b_i + 0.0330297796399062*v_GRID_b_r + 0.876104930717239*v_GRID_c_i + 0.032866807135454*v_GRID_c_r - v_POI_c_i
        struct[0].g[30,0] = -3.19498294936899*i_POI_a_i + 1.91805727135919*i_POI_a_r + 3.19498294936919*i_POI_b_i - 1.91805727135919*i_POI_b_r - 3.08738963687029*i_POImv_a_i + 11.6035242966407*i_POImv_a_r + 1.79198075607073*i_POImv_b_i + 10.1957014483333*i_POImv_b_r + 1.79198075607047*i_POImv_c_i + 10.1957014483333*i_POImv_c_r - 0.100328108879386*i_W1lv_a_i + 0.0318681229122168*i_W1lv_a_r + 0.100328108879391*i_W1lv_b_i - 0.0318681229122152*i_W1lv_b_r - 3.34841422289852*i_W1mv_a_i + 11.9248072396495*i_W1mv_a_r + 1.68849540047563*i_W1mv_b_i + 10.3248881664377*i_W1mv_b_r + 1.68849540047537*i_W1mv_c_i + 10.3248881664376*i_W1mv_c_r - 0.09928229701422*i_W2lv_a_i + 0.0305931173773991*i_W2lv_a_r + 0.0992822970142252*i_W2lv_b_i - 0.0305931173773975*i_W2lv_b_r - 3.26107847080994*i_W2mv_a_i + 11.81799648295*i_W2mv_a_r + 1.72332682544462*i_W2mv_b_i + 10.2820882609335*i_W2mv_b_r + 1.72332682544436*i_W2mv_c_i + 10.2820882609334*i_W2mv_c_r - 0.0982363113431474*i_W3lv_a_i + 0.0293176867112793*i_W3lv_a_r + 0.0982363113431526*i_W3lv_b_i - 0.0293176867112777*i_W3lv_b_r - 3.17407051442937*i_W3mv_a_i + 11.7109010138509*i_W3mv_a_r + 1.75782172888933*i_W3mv_b_i + 10.2390249864793*i_W3mv_b_r + 1.75782172888906*i_W3mv_c_i + 10.2390249864793*i_W3mv_c_r + 4.30097396370904e-5*v_GRID_a_i + 0.175093364713316*v_GRID_a_r - 4.30097396321211e-5*v_GRID_b_i - 0.175093364713324*v_GRID_b_r - 3.95107998085596e-17*v_GRID_c_i + 3.80502101367139e-17*v_GRID_c_r - v_W1mv_a_r
        struct[0].g[31,0] = 1.91805727135919*i_POI_a_i + 3.19498294936899*i_POI_a_r - 1.91805727135919*i_POI_b_i - 3.19498294936919*i_POI_b_r + 11.6035242966407*i_POImv_a_i + 3.08738963687029*i_POImv_a_r + 10.1957014483333*i_POImv_b_i - 1.79198075607073*i_POImv_b_r + 10.1957014483333*i_POImv_c_i - 1.79198075607047*i_POImv_c_r + 0.0318681229122168*i_W1lv_a_i + 0.100328108879386*i_W1lv_a_r - 0.0318681229122152*i_W1lv_b_i - 0.100328108879391*i_W1lv_b_r + 11.9248072396495*i_W1mv_a_i + 3.34841422289852*i_W1mv_a_r + 10.3248881664377*i_W1mv_b_i - 1.68849540047563*i_W1mv_b_r + 10.3248881664376*i_W1mv_c_i - 1.68849540047537*i_W1mv_c_r + 0.0305931173773991*i_W2lv_a_i + 0.09928229701422*i_W2lv_a_r - 0.0305931173773975*i_W2lv_b_i - 0.0992822970142252*i_W2lv_b_r + 11.81799648295*i_W2mv_a_i + 3.26107847080994*i_W2mv_a_r + 10.2820882609335*i_W2mv_b_i - 1.72332682544462*i_W2mv_b_r + 10.2820882609334*i_W2mv_c_i - 1.72332682544436*i_W2mv_c_r + 0.0293176867112793*i_W3lv_a_i + 0.0982363113431474*i_W3lv_a_r - 0.0293176867112777*i_W3lv_b_i - 0.0982363113431526*i_W3lv_b_r + 11.7109010138509*i_W3mv_a_i + 3.17407051442937*i_W3mv_a_r + 10.2390249864793*i_W3mv_b_i - 1.75782172888933*i_W3mv_b_r + 10.2390249864793*i_W3mv_c_i - 1.75782172888906*i_W3mv_c_r + 0.175093364713316*v_GRID_a_i - 4.30097396370904e-5*v_GRID_a_r - 0.175093364713324*v_GRID_b_i + 4.30097396321211e-5*v_GRID_b_r + 3.80502101367139e-17*v_GRID_c_i + 3.95107998085596e-17*v_GRID_c_r - v_W1mv_a_i
        struct[0].g[32,0] = -3.19498294936915*i_POI_b_i + 1.91805727135931*i_POI_b_r + 3.19498294936901*i_POI_c_i - 1.91805727135932*i_POI_c_r + 1.7919807560709*i_POImv_a_i + 10.1957014483335*i_POImv_a_r - 3.0873896368701*i_POImv_b_i + 11.603524296641*i_POImv_b_r + 1.79198075607072*i_POImv_c_i + 10.1957014483334*i_POImv_c_r - 0.100328108879391*i_W1lv_b_i + 0.0318681229122188*i_W1lv_b_r + 0.100328108879387*i_W1lv_c_i - 0.0318681229122203*i_W1lv_c_r + 1.68849540047579*i_W1mv_a_i + 10.3248881664379*i_W1mv_a_r - 3.34841422289833*i_W1mv_b_i + 11.9248072396498*i_W1mv_b_r + 1.68849540047561*i_W1mv_c_i + 10.3248881664378*i_W1mv_c_r - 0.099282297014225*i_W2lv_b_i + 0.0305931173774011*i_W2lv_b_r + 0.0992822970142214*i_W2lv_c_i - 0.0305931173774025*i_W2lv_c_r + 1.72332682544478*i_W2mv_a_i + 10.2820882609337*i_W2mv_a_r - 3.26107847080975*i_W2mv_b_i + 11.8179964829504*i_W2mv_b_r + 1.72332682544462*i_W2mv_c_i + 10.2820882609336*i_W2mv_c_r - 0.0982363113431523*i_W3lv_b_i + 0.0293176867112812*i_W3lv_b_r + 0.0982363113431489*i_W3lv_c_i - 0.0293176867112828*i_W3lv_c_r + 1.7578217288895*i_W3mv_a_i + 10.2390249864795*i_W3mv_a_r - 3.17407051442918*i_W3mv_b_i + 11.7109010138513*i_W3mv_b_r + 1.75782172888933*i_W3mv_c_i + 10.2390249864795*i_W3mv_c_r + 4.89578301787291e-18*v_GRID_a_i + 2.93583452294173e-18*v_GRID_a_r + 4.30097396379464e-5*v_GRID_b_i + 0.175093364713325*v_GRID_b_r - 4.30097396419938e-5*v_GRID_c_i - 0.17509336471332*v_GRID_c_r - v_W1mv_b_r
        struct[0].g[33,0] = 1.91805727135931*i_POI_b_i + 3.19498294936915*i_POI_b_r - 1.91805727135932*i_POI_c_i - 3.19498294936901*i_POI_c_r + 10.1957014483335*i_POImv_a_i - 1.7919807560709*i_POImv_a_r + 11.603524296641*i_POImv_b_i + 3.0873896368701*i_POImv_b_r + 10.1957014483334*i_POImv_c_i - 1.79198075607072*i_POImv_c_r + 0.0318681229122188*i_W1lv_b_i + 0.100328108879391*i_W1lv_b_r - 0.0318681229122203*i_W1lv_c_i - 0.100328108879387*i_W1lv_c_r + 10.3248881664379*i_W1mv_a_i - 1.68849540047579*i_W1mv_a_r + 11.9248072396498*i_W1mv_b_i + 3.34841422289833*i_W1mv_b_r + 10.3248881664378*i_W1mv_c_i - 1.68849540047561*i_W1mv_c_r + 0.0305931173774011*i_W2lv_b_i + 0.099282297014225*i_W2lv_b_r - 0.0305931173774025*i_W2lv_c_i - 0.0992822970142214*i_W2lv_c_r + 10.2820882609337*i_W2mv_a_i - 1.72332682544478*i_W2mv_a_r + 11.8179964829504*i_W2mv_b_i + 3.26107847080975*i_W2mv_b_r + 10.2820882609336*i_W2mv_c_i - 1.72332682544462*i_W2mv_c_r + 0.0293176867112812*i_W3lv_b_i + 0.0982363113431523*i_W3lv_b_r - 0.0293176867112828*i_W3lv_c_i - 0.0982363113431489*i_W3lv_c_r + 10.2390249864795*i_W3mv_a_i - 1.7578217288895*i_W3mv_a_r + 11.7109010138513*i_W3mv_b_i + 3.17407051442918*i_W3mv_b_r + 10.2390249864795*i_W3mv_c_i - 1.75782172888933*i_W3mv_c_r + 2.93583452294173e-18*v_GRID_a_i - 4.89578301787291e-18*v_GRID_a_r + 0.175093364713325*v_GRID_b_i - 4.30097396379464e-5*v_GRID_b_r - 0.17509336471332*v_GRID_c_i + 4.30097396419938e-5*v_GRID_c_r - v_W1mv_b_i
        struct[0].g[34,0] = 3.19498294936927*i_POI_a_i - 1.91805727135924*i_POI_a_r - 3.19498294936936*i_POI_c_i + 1.91805727135924*i_POI_c_r + 1.79198075607079*i_POImv_a_i + 10.1957014483334*i_POImv_a_r + 1.79198075607092*i_POImv_b_i + 10.1957014483334*i_POImv_b_r - 3.08738963687035*i_POImv_c_i + 11.6035242966408*i_POImv_c_r + 0.100328108879394*i_W1lv_a_i - 0.0318681229122163*i_W1lv_a_r - 0.100328108879396*i_W1lv_c_i + 0.0318681229122155*i_W1lv_c_r + 1.68849540047567*i_W1mv_a_i + 10.3248881664378*i_W1mv_a_r + 1.68849540047581*i_W1mv_b_i + 10.3248881664378*i_W1mv_b_r - 3.34841422289858*i_W1mv_c_i + 11.9248072396496*i_W1mv_c_r + 0.0992822970142275*i_W2lv_a_i - 0.0305931173773985*i_W2lv_a_r - 0.0992822970142302*i_W2lv_c_i + 0.0305931173773979*i_W2lv_c_r + 1.72332682544468*i_W2mv_a_i + 10.2820882609336*i_W2mv_a_r + 1.72332682544481*i_W2mv_b_i + 10.2820882609336*i_W2mv_b_r - 3.26107847081*i_W2mv_c_i + 11.8179964829502*i_W2mv_c_r + 0.098236311343155*i_W3lv_a_i - 0.0293176867112786*i_W3lv_a_r - 0.0982363113431575*i_W3lv_c_i + 0.0293176867112779*i_W3lv_c_r + 1.75782172888938*i_W3mv_a_i + 10.2390249864794*i_W3mv_a_r + 1.75782172888952*i_W3mv_b_i + 10.2390249864795*i_W3mv_b_r - 3.17407051442944*i_W3mv_c_i + 11.7109010138511*i_W3mv_c_r - 4.30097396325989e-5*v_GRID_a_i - 0.175093364713328*v_GRID_a_r + 3.10996770759589e-17*v_GRID_b_i - 4.08938066674102e-17*v_GRID_b_r + 4.30097396300291e-5*v_GRID_c_i + 0.175093364713332*v_GRID_c_r - v_W1mv_c_r
        struct[0].g[35,0] = -1.91805727135924*i_POI_a_i - 3.19498294936927*i_POI_a_r + 1.91805727135924*i_POI_c_i + 3.19498294936936*i_POI_c_r + 10.1957014483334*i_POImv_a_i - 1.79198075607079*i_POImv_a_r + 10.1957014483334*i_POImv_b_i - 1.79198075607092*i_POImv_b_r + 11.6035242966408*i_POImv_c_i + 3.08738963687035*i_POImv_c_r - 0.0318681229122163*i_W1lv_a_i - 0.100328108879394*i_W1lv_a_r + 0.0318681229122155*i_W1lv_c_i + 0.100328108879396*i_W1lv_c_r + 10.3248881664378*i_W1mv_a_i - 1.68849540047567*i_W1mv_a_r + 10.3248881664378*i_W1mv_b_i - 1.68849540047581*i_W1mv_b_r + 11.9248072396496*i_W1mv_c_i + 3.34841422289858*i_W1mv_c_r - 0.0305931173773985*i_W2lv_a_i - 0.0992822970142275*i_W2lv_a_r + 0.0305931173773979*i_W2lv_c_i + 0.0992822970142302*i_W2lv_c_r + 10.2820882609336*i_W2mv_a_i - 1.72332682544468*i_W2mv_a_r + 10.2820882609336*i_W2mv_b_i - 1.72332682544481*i_W2mv_b_r + 11.8179964829502*i_W2mv_c_i + 3.26107847081*i_W2mv_c_r - 0.0293176867112786*i_W3lv_a_i - 0.098236311343155*i_W3lv_a_r + 0.0293176867112779*i_W3lv_c_i + 0.0982363113431575*i_W3lv_c_r + 10.2390249864794*i_W3mv_a_i - 1.75782172888938*i_W3mv_a_r + 10.2390249864795*i_W3mv_b_i - 1.75782172888952*i_W3mv_b_r + 11.7109010138511*i_W3mv_c_i + 3.17407051442944*i_W3mv_c_r - 0.175093364713328*v_GRID_a_i + 4.30097396325989e-5*v_GRID_a_r - 4.08938066674102e-17*v_GRID_b_i - 3.10996770759589e-17*v_GRID_b_r + 0.175093364713332*v_GRID_c_i - 4.30097396300291e-5*v_GRID_c_r - v_W1mv_c_i
        struct[0].g[36,0] = -3.19498174821879*i_POI_a_i + 1.91804912205597*i_POI_a_r + 3.19498174821898*i_POI_b_i - 1.91804912205596*i_POI_b_r - 3.08755280951041*i_POImv_a_i + 11.60338540301*i_POImv_a_r + 1.79181314887337*i_POImv_b_i + 10.1955728673526*i_POImv_b_r + 1.79181314887312*i_POImv_c_i + 10.1955728673526*i_POImv_c_r - 0.0992822970142201*i_W1lv_a_i + 0.0305931173773991*i_W1lv_a_r + 0.0992822970142253*i_W1lv_b_i - 0.0305931173773975*i_W1lv_b_r - 3.26107847080993*i_W1mv_a_i + 11.81799648295*i_W1mv_a_r + 1.72332682544462*i_W1mv_b_i + 10.2820882609335*i_W1mv_b_r + 1.72332682544436*i_W1mv_c_i + 10.2820882609334*i_W1mv_c_r - 0.0992822101112667*i_W2lv_a_i + 0.030592904811748*i_W2lv_a_r + 0.0992822101112719*i_W2lv_b_i - 0.0305929048117464*i_W2lv_b_r - 3.26124236866395*i_W2mv_a_i + 11.8178541267502*i_W2mv_a_r + 1.72315856468247*i_W2mv_b_i + 10.2819565764585*i_W2mv_b_r + 1.72315856468222*i_W2mv_c_i + 10.2819565764584*i_W2mv_c_r - 0.0982362237268541*i_W3lv_a_i + 0.0293174777213171*i_W3lv_a_r + 0.0982362237268593*i_W3lv_b_i - 0.0293174777213155*i_W3lv_b_r - 3.17423405384011*i_W3mv_a_i + 11.7107603897954*i_W3mv_a_r + 1.75765379075767*i_W3mv_b_i + 10.2388948546334*i_W3mv_b_r + 1.75765379075741*i_W3mv_c_i + 10.2388948546333*i_W3mv_c_r + 4.27104401568212e-5*v_GRID_a_i + 0.175093119317178*v_GRID_a_r - 4.27104401518103e-5*v_GRID_b_i - 0.175093119317186*v_GRID_b_r - 3.95108094457718e-17*v_GRID_c_i + 3.8050089267765e-17*v_GRID_c_r - v_W2mv_a_r
        struct[0].g[37,0] = 1.91804912205597*i_POI_a_i + 3.19498174821879*i_POI_a_r - 1.91804912205596*i_POI_b_i - 3.19498174821898*i_POI_b_r + 11.60338540301*i_POImv_a_i + 3.08755280951041*i_POImv_a_r + 10.1955728673526*i_POImv_b_i - 1.79181314887337*i_POImv_b_r + 10.1955728673526*i_POImv_c_i - 1.79181314887312*i_POImv_c_r + 0.0305931173773991*i_W1lv_a_i + 0.0992822970142201*i_W1lv_a_r - 0.0305931173773975*i_W1lv_b_i - 0.0992822970142253*i_W1lv_b_r + 11.81799648295*i_W1mv_a_i + 3.26107847080993*i_W1mv_a_r + 10.2820882609335*i_W1mv_b_i - 1.72332682544462*i_W1mv_b_r + 10.2820882609334*i_W1mv_c_i - 1.72332682544436*i_W1mv_c_r + 0.030592904811748*i_W2lv_a_i + 0.0992822101112667*i_W2lv_a_r - 0.0305929048117464*i_W2lv_b_i - 0.0992822101112719*i_W2lv_b_r + 11.8178541267502*i_W2mv_a_i + 3.26124236866395*i_W2mv_a_r + 10.2819565764585*i_W2mv_b_i - 1.72315856468247*i_W2mv_b_r + 10.2819565764584*i_W2mv_c_i - 1.72315856468222*i_W2mv_c_r + 0.0293174777213171*i_W3lv_a_i + 0.0982362237268541*i_W3lv_a_r - 0.0293174777213155*i_W3lv_b_i - 0.0982362237268593*i_W3lv_b_r + 11.7107603897954*i_W3mv_a_i + 3.17423405384011*i_W3mv_a_r + 10.2388948546334*i_W3mv_b_i - 1.75765379075767*i_W3mv_b_r + 10.2388948546333*i_W3mv_c_i - 1.75765379075741*i_W3mv_c_r + 0.175093119317178*v_GRID_a_i - 4.27104401568212e-5*v_GRID_a_r - 0.175093119317186*v_GRID_b_i + 4.27104401518103e-5*v_GRID_b_r + 3.8050089267765e-17*v_GRID_c_i + 3.95108094457718e-17*v_GRID_c_r - v_W2mv_a_i
        struct[0].g[38,0] = -3.19498174821894*i_POI_b_i + 1.91804912205608*i_POI_b_r + 3.1949817482188*i_POI_c_i - 1.9180491220561*i_POI_c_r + 1.79181314887354*i_POImv_a_i + 10.1955728673528*i_POImv_a_r - 3.08755280951023*i_POImv_b_i + 11.6033854030104*i_POImv_b_r + 1.79181314887337*i_POImv_c_i + 10.1955728673527*i_POImv_c_r - 0.099282297014225*i_W1lv_b_i + 0.0305931173774011*i_W1lv_b_r + 0.0992822970142215*i_W1lv_c_i - 0.0305931173774025*i_W1lv_c_r + 1.72332682544478*i_W1mv_a_i + 10.2820882609337*i_W1mv_a_r - 3.26107847080975*i_W1mv_b_i + 11.8179964829504*i_W1mv_b_r + 1.72332682544461*i_W1mv_c_i + 10.2820882609336*i_W1mv_c_r - 0.0992822101112716*i_W2lv_b_i + 0.0305929048117499*i_W2lv_b_r + 0.0992822101112681*i_W2lv_c_i - 0.0305929048117514*i_W2lv_c_r + 1.72315856468264*i_W2mv_a_i + 10.2819565764587*i_W2mv_a_r - 3.26124236866376*i_W2mv_b_i + 11.8178541267506*i_W2mv_b_r + 1.72315856468247*i_W2mv_c_i + 10.2819565764586*i_W2mv_c_r - 0.098236223726859*i_W3lv_b_i + 0.029317477721319*i_W3lv_b_r + 0.0982362237268555*i_W3lv_c_i - 0.0293174777213205*i_W3lv_c_r + 1.75765379075784*i_W3mv_a_i + 10.2388948546336*i_W3mv_a_r - 3.17423405383993*i_W3mv_b_i + 11.7107603897957*i_W3mv_b_r + 1.75765379075766*i_W3mv_c_i + 10.2388948546335*i_W3mv_c_r + 4.8957711368811e-18*v_GRID_a_i + 2.93583877411288e-18*v_GRID_a_r + 4.27104401575523e-5*v_GRID_b_i + 0.175093119317187*v_GRID_b_r - 4.27104401616275e-5*v_GRID_c_i - 0.175093119317182*v_GRID_c_r - v_W2mv_b_r
        struct[0].g[39,0] = 1.91804912205608*i_POI_b_i + 3.19498174821894*i_POI_b_r - 1.9180491220561*i_POI_c_i - 3.1949817482188*i_POI_c_r + 10.1955728673528*i_POImv_a_i - 1.79181314887354*i_POImv_a_r + 11.6033854030104*i_POImv_b_i + 3.08755280951023*i_POImv_b_r + 10.1955728673527*i_POImv_c_i - 1.79181314887337*i_POImv_c_r + 0.0305931173774011*i_W1lv_b_i + 0.099282297014225*i_W1lv_b_r - 0.0305931173774025*i_W1lv_c_i - 0.0992822970142215*i_W1lv_c_r + 10.2820882609337*i_W1mv_a_i - 1.72332682544478*i_W1mv_a_r + 11.8179964829504*i_W1mv_b_i + 3.26107847080975*i_W1mv_b_r + 10.2820882609336*i_W1mv_c_i - 1.72332682544461*i_W1mv_c_r + 0.0305929048117499*i_W2lv_b_i + 0.0992822101112716*i_W2lv_b_r - 0.0305929048117514*i_W2lv_c_i - 0.0992822101112681*i_W2lv_c_r + 10.2819565764587*i_W2mv_a_i - 1.72315856468264*i_W2mv_a_r + 11.8178541267506*i_W2mv_b_i + 3.26124236866376*i_W2mv_b_r + 10.2819565764586*i_W2mv_c_i - 1.72315856468247*i_W2mv_c_r + 0.029317477721319*i_W3lv_b_i + 0.098236223726859*i_W3lv_b_r - 0.0293174777213205*i_W3lv_c_i - 0.0982362237268555*i_W3lv_c_r + 10.2388948546336*i_W3mv_a_i - 1.75765379075784*i_W3mv_a_r + 11.7107603897957*i_W3mv_b_i + 3.17423405383993*i_W3mv_b_r + 10.2388948546335*i_W3mv_c_i - 1.75765379075766*i_W3mv_c_r + 2.93583877411288e-18*v_GRID_a_i - 4.8957711368811e-18*v_GRID_a_r + 0.175093119317187*v_GRID_b_i - 4.27104401575523e-5*v_GRID_b_r - 0.175093119317182*v_GRID_c_i + 4.27104401616275e-5*v_GRID_c_r - v_W2mv_b_i
        struct[0].g[40,0] = 3.19498174821906*i_POI_a_i - 1.91804912205602*i_POI_a_r - 3.19498174821916*i_POI_c_i + 1.91804912205601*i_POI_c_r + 1.79181314887342*i_POImv_a_i + 10.1955728673527*i_POImv_a_r + 1.79181314887355*i_POImv_b_i + 10.1955728673527*i_POImv_b_r - 3.08755280951048*i_POImv_c_i + 11.6033854030101*i_POImv_c_r + 0.0992822970142276*i_W1lv_a_i - 0.0305931173773985*i_W1lv_a_r - 0.0992822970142302*i_W1lv_c_i + 0.0305931173773978*i_W1lv_c_r + 1.72332682544467*i_W1mv_a_i + 10.2820882609336*i_W1mv_a_r + 1.7233268254448*i_W1mv_b_i + 10.2820882609336*i_W1mv_b_r - 3.26107847081*i_W1mv_c_i + 11.8179964829501*i_W1mv_c_r + 0.0992822101112742*i_W2lv_a_i - 0.0305929048117474*i_W2lv_a_r - 0.0992822101112768*i_W2lv_c_i + 0.0305929048117467*i_W2lv_c_r + 1.72315856468253*i_W2mv_a_i + 10.2819565764586*i_W2mv_a_r + 1.72315856468266*i_W2mv_b_i + 10.2819565764586*i_W2mv_b_r - 3.26124236866401*i_W2mv_c_i + 11.8178541267503*i_W2mv_c_r + 0.0982362237268616*i_W3lv_a_i - 0.0293174777213164*i_W3lv_a_r - 0.0982362237268642*i_W3lv_c_i + 0.0293174777213157*i_W3lv_c_r + 1.75765379075772*i_W3mv_a_i + 10.2388948546335*i_W3mv_a_r + 1.75765379075786*i_W3mv_b_i + 10.2388948546335*i_W3mv_b_r - 3.17423405384018*i_W3mv_c_i + 11.7107603897955*i_W3mv_c_r - 4.2710440152288e-5*v_GRID_a_i - 0.175093119317191*v_GRID_a_r + 3.1099703364806e-17*v_GRID_b_i - 4.08936961867526e-17*v_GRID_b_r + 4.27104401497182e-5*v_GRID_c_i + 0.175093119317194*v_GRID_c_r - v_W2mv_c_r
        struct[0].g[41,0] = -1.91804912205602*i_POI_a_i - 3.19498174821906*i_POI_a_r + 1.91804912205601*i_POI_c_i + 3.19498174821916*i_POI_c_r + 10.1955728673527*i_POImv_a_i - 1.79181314887342*i_POImv_a_r + 10.1955728673527*i_POImv_b_i - 1.79181314887355*i_POImv_b_r + 11.6033854030101*i_POImv_c_i + 3.08755280951048*i_POImv_c_r - 0.0305931173773985*i_W1lv_a_i - 0.0992822970142276*i_W1lv_a_r + 0.0305931173773978*i_W1lv_c_i + 0.0992822970142302*i_W1lv_c_r + 10.2820882609336*i_W1mv_a_i - 1.72332682544467*i_W1mv_a_r + 10.2820882609336*i_W1mv_b_i - 1.7233268254448*i_W1mv_b_r + 11.8179964829501*i_W1mv_c_i + 3.26107847081*i_W1mv_c_r - 0.0305929048117474*i_W2lv_a_i - 0.0992822101112742*i_W2lv_a_r + 0.0305929048117467*i_W2lv_c_i + 0.0992822101112768*i_W2lv_c_r + 10.2819565764586*i_W2mv_a_i - 1.72315856468253*i_W2mv_a_r + 10.2819565764586*i_W2mv_b_i - 1.72315856468266*i_W2mv_b_r + 11.8178541267503*i_W2mv_c_i + 3.26124236866401*i_W2mv_c_r - 0.0293174777213164*i_W3lv_a_i - 0.0982362237268616*i_W3lv_a_r + 0.0293174777213157*i_W3lv_c_i + 0.0982362237268642*i_W3lv_c_r + 10.2388948546335*i_W3mv_a_i - 1.75765379075772*i_W3mv_a_r + 10.2388948546335*i_W3mv_b_i - 1.75765379075786*i_W3mv_b_r + 11.7107603897955*i_W3mv_c_i + 3.17423405384018*i_W3mv_c_r - 0.175093119317191*v_GRID_a_i + 4.2710440152288e-5*v_GRID_a_r - 4.08936961867526e-17*v_GRID_b_i - 3.1099703364806e-17*v_GRID_b_r + 0.175093119317194*v_GRID_c_i - 4.27104401497182e-5*v_GRID_c_r - v_W2mv_c_i
        struct[0].g[42,0] = -3.19497814474368*i_POI_a_i + 1.91802467417325*i_POI_a_r + 3.19497814474388*i_POI_b_i - 1.91802467417324*i_POI_b_r - 3.08804231916211*i_POImv_a_i + 11.6029687203683*i_POImv_a_r + 1.79131033552716*i_POImv_b_i + 10.1951871226168*i_POImv_b_r + 1.7913103355269*i_POImv_c_i + 10.1951871226167*i_POImv_c_r - 0.0982363113431474*i_W1lv_a_i + 0.0293176867112792*i_W1lv_a_r + 0.0982363113431526*i_W1lv_b_i - 0.0293176867112776*i_W1lv_b_r - 3.17407051442937*i_W1mv_a_i + 11.7109010138509*i_W1mv_a_r + 1.75782172888933*i_W1mv_b_i + 10.2390249864793*i_W1mv_b_r + 1.75782172888907*i_W1mv_c_i + 10.2390249864793*i_W1mv_c_r - 0.0982362237268541*i_W2lv_a_i + 0.029317477721317*i_W2lv_a_r + 0.0982362237268593*i_W2lv_b_i - 0.0293174777213155*i_W2lv_b_r - 3.17423405384011*i_W2mv_a_i + 11.7107603897953*i_W2mv_a_r + 1.75765379075767*i_W2mv_b_i + 10.2388948546334*i_W2mv_b_r + 1.75765379075741*i_W2mv_c_i + 10.2388948546333*i_W2mv_c_r - 0.0982359608775054*i_W3lv_a_i + 0.0293168507523162*i_W3lv_a_r + 0.0982359608775105*i_W3lv_b_i - 0.0293168507523146*i_W3lv_b_r - 3.17472466374498*i_W3mv_a_i + 11.7103385159093*i_W3mv_a_r + 1.75714998466652*i_W3mv_b_i + 10.2385044573318*i_W3mv_b_r + 1.75714998466626*i_W3mv_c_i + 10.2385044573317*i_W3mv_c_r + 4.18125433939492e-5*v_GRID_a_i + 0.17509238312843*v_GRID_a_r - 4.18125433889522e-5*v_GRID_b_i - 0.175092383128438*v_GRID_b_r - 3.9510838356968e-17*v_GRID_c_i + 3.80497266612242e-17*v_GRID_c_r - v_W3mv_a_r
        struct[0].g[43,0] = 1.91802467417325*i_POI_a_i + 3.19497814474368*i_POI_a_r - 1.91802467417324*i_POI_b_i - 3.19497814474388*i_POI_b_r + 11.6029687203683*i_POImv_a_i + 3.08804231916211*i_POImv_a_r + 10.1951871226168*i_POImv_b_i - 1.79131033552716*i_POImv_b_r + 10.1951871226167*i_POImv_c_i - 1.7913103355269*i_POImv_c_r + 0.0293176867112792*i_W1lv_a_i + 0.0982363113431474*i_W1lv_a_r - 0.0293176867112776*i_W1lv_b_i - 0.0982363113431526*i_W1lv_b_r + 11.7109010138509*i_W1mv_a_i + 3.17407051442937*i_W1mv_a_r + 10.2390249864793*i_W1mv_b_i - 1.75782172888933*i_W1mv_b_r + 10.2390249864793*i_W1mv_c_i - 1.75782172888907*i_W1mv_c_r + 0.029317477721317*i_W2lv_a_i + 0.0982362237268541*i_W2lv_a_r - 0.0293174777213155*i_W2lv_b_i - 0.0982362237268593*i_W2lv_b_r + 11.7107603897953*i_W2mv_a_i + 3.17423405384011*i_W2mv_a_r + 10.2388948546334*i_W2mv_b_i - 1.75765379075767*i_W2mv_b_r + 10.2388948546333*i_W2mv_c_i - 1.75765379075741*i_W2mv_c_r + 0.0293168507523162*i_W3lv_a_i + 0.0982359608775054*i_W3lv_a_r - 0.0293168507523146*i_W3lv_b_i - 0.0982359608775105*i_W3lv_b_r + 11.7103385159093*i_W3mv_a_i + 3.17472466374498*i_W3mv_a_r + 10.2385044573318*i_W3mv_b_i - 1.75714998466652*i_W3mv_b_r + 10.2385044573317*i_W3mv_c_i - 1.75714998466626*i_W3mv_c_r + 0.17509238312843*v_GRID_a_i - 4.18125433939492e-5*v_GRID_a_r - 0.175092383128438*v_GRID_b_i + 4.18125433889522e-5*v_GRID_b_r + 3.80497266612242e-17*v_GRID_c_i + 3.9510838356968e-17*v_GRID_c_r - v_W3mv_a_i
        struct[0].g[44,0] = -3.19497814474384*i_POI_b_i + 1.91802467417336*i_POI_b_r + 3.1949781447437*i_POI_c_i - 1.91802467417338*i_POI_c_r + 1.79131033552732*i_POImv_a_i + 10.1951871226169*i_POImv_a_r - 3.08804231916193*i_POImv_b_i + 11.6029687203686*i_POImv_b_r + 1.79131033552715*i_POImv_c_i + 10.1951871226169*i_POImv_c_r - 0.0982363113431524*i_W1lv_b_i + 0.0293176867112812*i_W1lv_b_r + 0.0982363113431489*i_W1lv_c_i - 0.0293176867112826*i_W1lv_c_r + 1.7578217288895*i_W1mv_a_i + 10.2390249864795*i_W1mv_a_r - 3.17407051442919*i_W1mv_b_i + 11.7109010138513*i_W1mv_b_r + 1.75782172888932*i_W1mv_c_i + 10.2390249864794*i_W1mv_c_r - 0.098236223726859*i_W2lv_b_i + 0.029317477721319*i_W2lv_b_r + 0.0982362237268555*i_W2lv_c_i - 0.0293174777213204*i_W2lv_c_r + 1.75765379075784*i_W2mv_a_i + 10.2388948546336*i_W2mv_a_r - 3.17423405383993*i_W2mv_b_i + 11.7107603897957*i_W2mv_b_r + 1.75765379075766*i_W2mv_c_i + 10.2388948546335*i_W2mv_c_r - 0.0982359608775103*i_W3lv_b_i + 0.0293168507523179*i_W3lv_b_r + 0.0982359608775067*i_W3lv_c_i - 0.0293168507523194*i_W3lv_c_r + 1.75714998466669*i_W3mv_a_i + 10.238504457332*i_W3mv_a_r - 3.1747246637448*i_W3mv_b_i + 11.7103385159096*i_W3mv_b_r + 1.75714998466651*i_W3mv_c_i + 10.2385044573319*i_W3mv_c_r + 4.89573549392443e-18*v_GRID_a_i + 2.93585152757382e-18*v_GRID_a_r + 4.18125433946109e-5*v_GRID_b_i + 0.175092383128439*v_GRID_b_r - 4.18125433986861e-5*v_GRID_c_i - 0.175092383128434*v_GRID_c_r - v_W3mv_b_r
        struct[0].g[45,0] = 1.91802467417336*i_POI_b_i + 3.19497814474384*i_POI_b_r - 1.91802467417338*i_POI_c_i - 3.1949781447437*i_POI_c_r + 10.1951871226169*i_POImv_a_i - 1.79131033552732*i_POImv_a_r + 11.6029687203686*i_POImv_b_i + 3.08804231916193*i_POImv_b_r + 10.1951871226169*i_POImv_c_i - 1.79131033552715*i_POImv_c_r + 0.0293176867112812*i_W1lv_b_i + 0.0982363113431524*i_W1lv_b_r - 0.0293176867112826*i_W1lv_c_i - 0.0982363113431489*i_W1lv_c_r + 10.2390249864795*i_W1mv_a_i - 1.7578217288895*i_W1mv_a_r + 11.7109010138513*i_W1mv_b_i + 3.17407051442919*i_W1mv_b_r + 10.2390249864794*i_W1mv_c_i - 1.75782172888932*i_W1mv_c_r + 0.029317477721319*i_W2lv_b_i + 0.098236223726859*i_W2lv_b_r - 0.0293174777213204*i_W2lv_c_i - 0.0982362237268555*i_W2lv_c_r + 10.2388948546336*i_W2mv_a_i - 1.75765379075784*i_W2mv_a_r + 11.7107603897957*i_W2mv_b_i + 3.17423405383993*i_W2mv_b_r + 10.2388948546335*i_W2mv_c_i - 1.75765379075766*i_W2mv_c_r + 0.0293168507523179*i_W3lv_b_i + 0.0982359608775103*i_W3lv_b_r - 0.0293168507523194*i_W3lv_c_i - 0.0982359608775067*i_W3lv_c_r + 10.238504457332*i_W3mv_a_i - 1.75714998466669*i_W3mv_a_r + 11.7103385159096*i_W3mv_b_i + 3.1747246637448*i_W3mv_b_r + 10.2385044573319*i_W3mv_c_i - 1.75714998466651*i_W3mv_c_r + 2.93585152757382e-18*v_GRID_a_i - 4.89573549392443e-18*v_GRID_a_r + 0.175092383128439*v_GRID_b_i - 4.18125433946109e-5*v_GRID_b_r - 0.175092383128434*v_GRID_c_i + 4.18125433986861e-5*v_GRID_c_r - v_W3mv_b_i
        struct[0].g[46,0] = 3.19497814474395*i_POI_a_i - 1.9180246741733*i_POI_a_r - 3.19497814474405*i_POI_c_i + 1.91802467417329*i_POI_c_r + 1.79131033552721*i_POImv_a_i + 10.1951871226168*i_POImv_a_r + 1.79131033552733*i_POImv_b_i + 10.1951871226169*i_POImv_b_r - 3.08804231916218*i_POImv_c_i + 11.6029687203684*i_POImv_c_r + 0.098236311343155*i_W1lv_a_i - 0.0293176867112787*i_W1lv_a_r - 0.0982363113431576*i_W1lv_c_i + 0.0293176867112779*i_W1lv_c_r + 1.75782172888938*i_W1mv_a_i + 10.2390249864794*i_W1mv_a_r + 1.75782172888951*i_W1mv_b_i + 10.2390249864795*i_W1mv_b_r - 3.17407051442944*i_W1mv_c_i + 11.7109010138511*i_W1mv_c_r + 0.0982362237268616*i_W2lv_a_i - 0.0293174777213165*i_W2lv_a_r - 0.0982362237268642*i_W2lv_c_i + 0.0293174777213157*i_W2lv_c_r + 1.75765379075772*i_W2mv_a_i + 10.2388948546335*i_W2mv_a_r + 1.75765379075785*i_W2mv_b_i + 10.2388948546335*i_W2mv_b_r - 3.17423405384018*i_W2mv_c_i + 11.7107603897955*i_W2mv_c_r + 0.0982359608775129*i_W3lv_a_i - 0.0293168507523155*i_W3lv_a_r - 0.0982359608775155*i_W3lv_c_i + 0.0293168507523147*i_W3lv_c_r + 1.75714998466657*i_W3mv_a_i + 10.2385044573319*i_W3mv_a_r + 1.7571499846667*i_W3mv_b_i + 10.2385044573319*i_W3mv_b_r - 3.17472466374505*i_W3mv_c_i + 11.7103385159094*i_W3mv_c_r - 4.1812543389416e-5*v_GRID_a_i - 0.175092383128442*v_GRID_a_r + 3.1099782230896e-17*v_GRID_b_i - 4.08933647449996e-17*v_GRID_b_r + 4.18125433868462e-5*v_GRID_c_i + 0.175092383128446*v_GRID_c_r - v_W3mv_c_r
        struct[0].g[47,0] = -1.9180246741733*i_POI_a_i - 3.19497814474395*i_POI_a_r + 1.91802467417329*i_POI_c_i + 3.19497814474405*i_POI_c_r + 10.1951871226168*i_POImv_a_i - 1.79131033552721*i_POImv_a_r + 10.1951871226169*i_POImv_b_i - 1.79131033552733*i_POImv_b_r + 11.6029687203684*i_POImv_c_i + 3.08804231916218*i_POImv_c_r - 0.0293176867112787*i_W1lv_a_i - 0.098236311343155*i_W1lv_a_r + 0.0293176867112779*i_W1lv_c_i + 0.0982363113431576*i_W1lv_c_r + 10.2390249864794*i_W1mv_a_i - 1.75782172888938*i_W1mv_a_r + 10.2390249864795*i_W1mv_b_i - 1.75782172888951*i_W1mv_b_r + 11.7109010138511*i_W1mv_c_i + 3.17407051442944*i_W1mv_c_r - 0.0293174777213165*i_W2lv_a_i - 0.0982362237268616*i_W2lv_a_r + 0.0293174777213157*i_W2lv_c_i + 0.0982362237268642*i_W2lv_c_r + 10.2388948546335*i_W2mv_a_i - 1.75765379075772*i_W2mv_a_r + 10.2388948546335*i_W2mv_b_i - 1.75765379075785*i_W2mv_b_r + 11.7107603897955*i_W2mv_c_i + 3.17423405384018*i_W2mv_c_r - 0.0293168507523155*i_W3lv_a_i - 0.0982359608775129*i_W3lv_a_r + 0.0293168507523147*i_W3lv_c_i + 0.0982359608775155*i_W3lv_c_r + 10.2385044573319*i_W3mv_a_i - 1.75714998466657*i_W3mv_a_r + 10.2385044573319*i_W3mv_b_i - 1.7571499846667*i_W3mv_b_r + 11.7103385159094*i_W3mv_c_i + 3.17472466374505*i_W3mv_c_r - 0.175092383128442*v_GRID_a_i + 4.1812543389416e-5*v_GRID_a_r - 4.08933647449996e-17*v_GRID_b_i - 3.1099782230896e-17*v_GRID_b_r + 0.175092383128446*v_GRID_c_i - 4.18125433868462e-5*v_GRID_c_r - v_W3mv_c_i
        struct[0].g[48,0] = i_W1lv_a_i*v_W1lv_a_i + i_W1lv_a_r*v_W1lv_a_r - p_W1lv_a
        struct[0].g[49,0] = i_W1lv_b_i*v_W1lv_b_i + i_W1lv_b_r*v_W1lv_b_r - p_W1lv_b
        struct[0].g[50,0] = i_W1lv_c_i*v_W1lv_c_i + i_W1lv_c_r*v_W1lv_c_r - p_W1lv_c
        struct[0].g[51,0] = -i_W1lv_a_i*v_W1lv_a_r + i_W1lv_a_r*v_W1lv_a_i - q_W1lv_a
        struct[0].g[52,0] = -i_W1lv_b_i*v_W1lv_b_r + i_W1lv_b_r*v_W1lv_b_i - q_W1lv_b
        struct[0].g[53,0] = -i_W1lv_c_i*v_W1lv_c_r + i_W1lv_c_r*v_W1lv_c_i - q_W1lv_c
        struct[0].g[54,0] = i_W2lv_a_i*v_W2lv_a_i + i_W2lv_a_r*v_W2lv_a_r - p_W2lv_a
        struct[0].g[55,0] = i_W2lv_b_i*v_W2lv_b_i + i_W2lv_b_r*v_W2lv_b_r - p_W2lv_b
        struct[0].g[56,0] = i_W2lv_c_i*v_W2lv_c_i + i_W2lv_c_r*v_W2lv_c_r - p_W2lv_c
        struct[0].g[57,0] = -i_W2lv_a_i*v_W2lv_a_r + i_W2lv_a_r*v_W2lv_a_i - q_W2lv_a
        struct[0].g[58,0] = -i_W2lv_b_i*v_W2lv_b_r + i_W2lv_b_r*v_W2lv_b_i - q_W2lv_b
        struct[0].g[59,0] = -i_W2lv_c_i*v_W2lv_c_r + i_W2lv_c_r*v_W2lv_c_i - q_W2lv_c
        struct[0].g[60,0] = i_W3lv_a_i*v_W3lv_a_i + i_W3lv_a_r*v_W3lv_a_r - p_W3lv_a
        struct[0].g[61,0] = i_W3lv_b_i*v_W3lv_b_i + i_W3lv_b_r*v_W3lv_b_r - p_W3lv_b
        struct[0].g[62,0] = i_W3lv_c_i*v_W3lv_c_i + i_W3lv_c_r*v_W3lv_c_r - p_W3lv_c
        struct[0].g[63,0] = -i_W3lv_a_i*v_W3lv_a_r + i_W3lv_a_r*v_W3lv_a_i - q_W3lv_a
        struct[0].g[64,0] = -i_W3lv_b_i*v_W3lv_b_r + i_W3lv_b_r*v_W3lv_b_i - q_W3lv_b
        struct[0].g[65,0] = -i_W3lv_c_i*v_W3lv_c_r + i_W3lv_c_r*v_W3lv_c_i - q_W3lv_c
        struct[0].g[66,0] = i_POImv_a_i*v_POImv_a_i + i_POImv_a_r*v_POImv_a_r - p_POImv_a
        struct[0].g[67,0] = i_POImv_b_i*v_POImv_b_i + i_POImv_b_r*v_POImv_b_r - p_POImv_b
        struct[0].g[68,0] = i_POImv_c_i*v_POImv_c_i + i_POImv_c_r*v_POImv_c_r - p_POImv_c
        struct[0].g[69,0] = -i_POImv_a_i*v_POImv_a_r + i_POImv_a_r*v_POImv_a_i - q_POImv_a
        struct[0].g[70,0] = -i_POImv_b_i*v_POImv_b_r + i_POImv_b_r*v_POImv_b_i - q_POImv_b
        struct[0].g[71,0] = -i_POImv_c_i*v_POImv_c_r + i_POImv_c_r*v_POImv_c_i - q_POImv_c
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_GRID_a_i**2 + v_GRID_a_r**2)**0.5
        struct[0].h[1,0] = (v_GRID_b_i**2 + v_GRID_b_r**2)**0.5
        struct[0].h[2,0] = (v_GRID_c_i**2 + v_GRID_c_r**2)**0.5
        struct[0].h[3,0] = (v_W1lv_a_i**2 + v_W1lv_a_r**2)**0.5
        struct[0].h[4,0] = (v_W1lv_b_i**2 + v_W1lv_b_r**2)**0.5
        struct[0].h[5,0] = (v_W1lv_c_i**2 + v_W1lv_c_r**2)**0.5
        struct[0].h[6,0] = (v_W2lv_a_i**2 + v_W2lv_a_r**2)**0.5
        struct[0].h[7,0] = (v_W2lv_b_i**2 + v_W2lv_b_r**2)**0.5
        struct[0].h[8,0] = (v_W2lv_c_i**2 + v_W2lv_c_r**2)**0.5
        struct[0].h[9,0] = (v_W3lv_a_i**2 + v_W3lv_a_r**2)**0.5
        struct[0].h[10,0] = (v_W3lv_b_i**2 + v_W3lv_b_r**2)**0.5
        struct[0].h[11,0] = (v_W3lv_c_i**2 + v_W3lv_c_r**2)**0.5
        struct[0].h[12,0] = (v_POImv_a_i**2 + v_POImv_a_r**2)**0.5
        struct[0].h[13,0] = (v_POImv_b_i**2 + v_POImv_b_r**2)**0.5
        struct[0].h[14,0] = (v_POImv_c_i**2 + v_POImv_c_r**2)**0.5
        struct[0].h[15,0] = (v_POI_a_i**2 + v_POI_a_r**2)**0.5
        struct[0].h[16,0] = (v_POI_b_i**2 + v_POI_b_r**2)**0.5
        struct[0].h[17,0] = (v_POI_c_i**2 + v_POI_c_r**2)**0.5
        struct[0].h[18,0] = (v_W1mv_a_i**2 + v_W1mv_a_r**2)**0.5
        struct[0].h[19,0] = (v_W1mv_b_i**2 + v_W1mv_b_r**2)**0.5
        struct[0].h[20,0] = (v_W1mv_c_i**2 + v_W1mv_c_r**2)**0.5
        struct[0].h[21,0] = (v_W2mv_a_i**2 + v_W2mv_a_r**2)**0.5
        struct[0].h[22,0] = (v_W2mv_b_i**2 + v_W2mv_b_r**2)**0.5
        struct[0].h[23,0] = (v_W2mv_c_i**2 + v_W2mv_c_r**2)**0.5
        struct[0].h[24,0] = (v_W3mv_a_i**2 + v_W3mv_a_r**2)**0.5
        struct[0].h[25,0] = (v_W3mv_b_i**2 + v_W3mv_b_r**2)**0.5
        struct[0].h[26,0] = (v_W3mv_c_i**2 + v_W3mv_c_r**2)**0.5
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -1

    if mode == 11:



        struct[0].Gy_ini[0,0] = -1
        struct[0].Gy_ini[0,48] = 0.00317393578459360
        struct[0].Gy_ini[0,49] = -0.0154231877861473
        struct[0].Gy_ini[0,50] = -0.000634767892296793
        struct[0].Gy_ini[0,51] = 0.00199839389307364
        struct[0].Gy_ini[0,52] = -0.000634767892296814
        struct[0].Gy_ini[0,53] = 0.00199839389307368
        struct[0].Gy_ini[0,54] = 0.00121874317417018
        struct[0].Gy_ini[0,55] = -0.00395512560257793
        struct[0].Gy_ini[0,56] = -0.000609371587085080
        struct[0].Gy_ini[0,57] = 0.00197756280128894
        struct[0].Gy_ini[0,58] = -0.000609371587085102
        struct[0].Gy_ini[0,59] = 0.00197756280128899
        struct[0].Gy_ini[0,60] = 0.00116793362771941
        struct[0].Gy_ini[0,61] = -0.00391345649507333
        struct[0].Gy_ini[0,62] = -0.000583966813859693
        struct[0].Gy_ini[0,63] = 0.00195672824753664
        struct[0].Gy_ini[0,64] = -0.000583966813859714
        struct[0].Gy_ini[0,65] = 0.00195672824753669
        struct[0].Gy_ini[0,66] = 0.0280418380652321
        struct[0].Gy_ini[0,67] = -0.0971901504394946
        struct[0].Gy_ini[0,70] = -0.0280418380652375
        struct[0].Gy_ini[0,71] = 0.0971901504394895
        struct[0].Gy_ini[1,1] = -1
        struct[0].Gy_ini[1,48] = 0.0154231877861473
        struct[0].Gy_ini[1,49] = 0.00317393578459360
        struct[0].Gy_ini[1,50] = -0.00199839389307364
        struct[0].Gy_ini[1,51] = -0.000634767892296793
        struct[0].Gy_ini[1,52] = -0.00199839389307368
        struct[0].Gy_ini[1,53] = -0.000634767892296814
        struct[0].Gy_ini[1,54] = 0.00395512560257793
        struct[0].Gy_ini[1,55] = 0.00121874317417018
        struct[0].Gy_ini[1,56] = -0.00197756280128894
        struct[0].Gy_ini[1,57] = -0.000609371587085080
        struct[0].Gy_ini[1,58] = -0.00197756280128899
        struct[0].Gy_ini[1,59] = -0.000609371587085102
        struct[0].Gy_ini[1,60] = 0.00391345649507333
        struct[0].Gy_ini[1,61] = 0.00116793362771941
        struct[0].Gy_ini[1,62] = -0.00195672824753664
        struct[0].Gy_ini[1,63] = -0.000583966813859693
        struct[0].Gy_ini[1,64] = -0.00195672824753669
        struct[0].Gy_ini[1,65] = -0.000583966813859714
        struct[0].Gy_ini[1,66] = 0.0971901504394946
        struct[0].Gy_ini[1,67] = 0.0280418380652321
        struct[0].Gy_ini[1,70] = -0.0971901504394895
        struct[0].Gy_ini[1,71] = -0.0280418380652375
        struct[0].Gy_ini[2,2] = -1
        struct[0].Gy_ini[2,48] = -0.000634767892296777
        struct[0].Gy_ini[2,49] = 0.00199839389307366
        struct[0].Gy_ini[2,50] = 0.00317393578459362
        struct[0].Gy_ini[2,51] = -0.0154231877861474
        struct[0].Gy_ini[2,52] = -0.000634767892296846
        struct[0].Gy_ini[2,53] = 0.00199839389307372
        struct[0].Gy_ini[2,54] = -0.000609371587085066
        struct[0].Gy_ini[2,55] = 0.00197756280128896
        struct[0].Gy_ini[2,56] = 0.00121874317417020
        struct[0].Gy_ini[2,57] = -0.00395512560257798
        struct[0].Gy_ini[2,58] = -0.000609371587085134
        struct[0].Gy_ini[2,59] = 0.00197756280128902
        struct[0].Gy_ini[2,60] = -0.000583966813859679
        struct[0].Gy_ini[2,61] = 0.00195672824753666
        struct[0].Gy_ini[2,62] = 0.00116793362771943
        struct[0].Gy_ini[2,63] = -0.00391345649507338
        struct[0].Gy_ini[2,64] = -0.000583966813859747
        struct[0].Gy_ini[2,65] = 0.00195672824753672
        struct[0].Gy_ini[2,66] = -0.0280418380652299
        struct[0].Gy_ini[2,67] = 0.0971901504394967
        struct[0].Gy_ini[2,68] = 0.0280418380652406
        struct[0].Gy_ini[2,69] = -0.0971901504394895
        struct[0].Gy_ini[3,3] = -1
        struct[0].Gy_ini[3,48] = -0.00199839389307366
        struct[0].Gy_ini[3,49] = -0.000634767892296777
        struct[0].Gy_ini[3,50] = 0.0154231877861474
        struct[0].Gy_ini[3,51] = 0.00317393578459362
        struct[0].Gy_ini[3,52] = -0.00199839389307372
        struct[0].Gy_ini[3,53] = -0.000634767892296846
        struct[0].Gy_ini[3,54] = -0.00197756280128896
        struct[0].Gy_ini[3,55] = -0.000609371587085066
        struct[0].Gy_ini[3,56] = 0.00395512560257798
        struct[0].Gy_ini[3,57] = 0.00121874317417020
        struct[0].Gy_ini[3,58] = -0.00197756280128902
        struct[0].Gy_ini[3,59] = -0.000609371587085134
        struct[0].Gy_ini[3,60] = -0.00195672824753666
        struct[0].Gy_ini[3,61] = -0.000583966813859679
        struct[0].Gy_ini[3,62] = 0.00391345649507338
        struct[0].Gy_ini[3,63] = 0.00116793362771943
        struct[0].Gy_ini[3,64] = -0.00195672824753672
        struct[0].Gy_ini[3,65] = -0.000583966813859747
        struct[0].Gy_ini[3,66] = -0.0971901504394967
        struct[0].Gy_ini[3,67] = -0.0280418380652299
        struct[0].Gy_ini[3,68] = 0.0971901504394895
        struct[0].Gy_ini[3,69] = 0.0280418380652406
        struct[0].Gy_ini[4,4] = -1
        struct[0].Gy_ini[4,48] = -0.000634767892296827
        struct[0].Gy_ini[4,49] = 0.00199839389307367
        struct[0].Gy_ini[4,50] = -0.000634767892296831
        struct[0].Gy_ini[4,51] = 0.00199839389307374
        struct[0].Gy_ini[4,52] = 0.00317393578459366
        struct[0].Gy_ini[4,53] = -0.0154231877861474
        struct[0].Gy_ini[4,54] = -0.000609371587085114
        struct[0].Gy_ini[4,55] = 0.00197756280128897
        struct[0].Gy_ini[4,56] = -0.000609371587085120
        struct[0].Gy_ini[4,57] = 0.00197756280128904
        struct[0].Gy_ini[4,58] = 0.00121874317417024
        struct[0].Gy_ini[4,59] = -0.00395512560257801
        struct[0].Gy_ini[4,60] = -0.000583966813859728
        struct[0].Gy_ini[4,61] = 0.00195672824753667
        struct[0].Gy_ini[4,62] = -0.000583966813859733
        struct[0].Gy_ini[4,63] = 0.00195672824753674
        struct[0].Gy_ini[4,64] = 0.00116793362771946
        struct[0].Gy_ini[4,65] = -0.00391345649507341
        struct[0].Gy_ini[4,68] = -0.0280418380652387
        struct[0].Gy_ini[4,69] = 0.0971901504394933
        struct[0].Gy_ini[4,70] = 0.0280418380652339
        struct[0].Gy_ini[4,71] = -0.0971901504394944
        struct[0].Gy_ini[5,5] = -1
        struct[0].Gy_ini[5,48] = -0.00199839389307367
        struct[0].Gy_ini[5,49] = -0.000634767892296827
        struct[0].Gy_ini[5,50] = -0.00199839389307374
        struct[0].Gy_ini[5,51] = -0.000634767892296831
        struct[0].Gy_ini[5,52] = 0.0154231877861474
        struct[0].Gy_ini[5,53] = 0.00317393578459366
        struct[0].Gy_ini[5,54] = -0.00197756280128897
        struct[0].Gy_ini[5,55] = -0.000609371587085114
        struct[0].Gy_ini[5,56] = -0.00197756280128904
        struct[0].Gy_ini[5,57] = -0.000609371587085120
        struct[0].Gy_ini[5,58] = 0.00395512560257801
        struct[0].Gy_ini[5,59] = 0.00121874317417024
        struct[0].Gy_ini[5,60] = -0.00195672824753667
        struct[0].Gy_ini[5,61] = -0.000583966813859728
        struct[0].Gy_ini[5,62] = -0.00195672824753674
        struct[0].Gy_ini[5,63] = -0.000583966813859733
        struct[0].Gy_ini[5,64] = 0.00391345649507341
        struct[0].Gy_ini[5,65] = 0.00116793362771946
        struct[0].Gy_ini[5,68] = -0.0971901504394933
        struct[0].Gy_ini[5,69] = -0.0280418380652387
        struct[0].Gy_ini[5,70] = 0.0971901504394944
        struct[0].Gy_ini[5,71] = 0.0280418380652339
        struct[0].Gy_ini[6,6] = -1
        struct[0].Gy_ini[6,48] = 0.00121874317417018
        struct[0].Gy_ini[6,49] = -0.00395512560257793
        struct[0].Gy_ini[6,50] = -0.000609371587085081
        struct[0].Gy_ini[6,51] = 0.00197756280128894
        struct[0].Gy_ini[6,52] = -0.000609371587085101
        struct[0].Gy_ini[6,53] = 0.00197756280128899
        struct[0].Gy_ini[6,54] = 0.00312313470615651
        struct[0].Gy_ini[6,55] = -0.0153815221406103
        struct[0].Gy_ini[6,56] = -0.000609367353078243
        struct[0].Gy_ini[6,57] = 0.00197756107030514
        struct[0].Gy_ini[6,58] = -0.000609367353078263
        struct[0].Gy_ini[6,59] = 0.00197756107030519
        struct[0].Gy_ini[6,60] = 0.00116792530215105
        struct[0].Gy_ini[6,61] = -0.00391345300468828
        struct[0].Gy_ini[6,62] = -0.000583962651075517
        struct[0].Gy_ini[6,63] = 0.00195672650234412
        struct[0].Gy_ini[6,64] = -0.000583962651075537
        struct[0].Gy_ini[6,65] = 0.00195672650234417
        struct[0].Gy_ini[6,66] = 0.0280416326518444
        struct[0].Gy_ini[6,67] = -0.0971900621093923
        struct[0].Gy_ini[6,70] = -0.0280416326518497
        struct[0].Gy_ini[6,71] = 0.0971900621093875
        struct[0].Gy_ini[7,7] = -1
        struct[0].Gy_ini[7,48] = 0.00395512560257793
        struct[0].Gy_ini[7,49] = 0.00121874317417018
        struct[0].Gy_ini[7,50] = -0.00197756280128894
        struct[0].Gy_ini[7,51] = -0.000609371587085081
        struct[0].Gy_ini[7,52] = -0.00197756280128899
        struct[0].Gy_ini[7,53] = -0.000609371587085101
        struct[0].Gy_ini[7,54] = 0.0153815221406103
        struct[0].Gy_ini[7,55] = 0.00312313470615651
        struct[0].Gy_ini[7,56] = -0.00197756107030514
        struct[0].Gy_ini[7,57] = -0.000609367353078243
        struct[0].Gy_ini[7,58] = -0.00197756107030519
        struct[0].Gy_ini[7,59] = -0.000609367353078263
        struct[0].Gy_ini[7,60] = 0.00391345300468828
        struct[0].Gy_ini[7,61] = 0.00116792530215105
        struct[0].Gy_ini[7,62] = -0.00195672650234412
        struct[0].Gy_ini[7,63] = -0.000583962651075517
        struct[0].Gy_ini[7,64] = -0.00195672650234417
        struct[0].Gy_ini[7,65] = -0.000583962651075537
        struct[0].Gy_ini[7,66] = 0.0971900621093923
        struct[0].Gy_ini[7,67] = 0.0280416326518444
        struct[0].Gy_ini[7,70] = -0.0971900621093875
        struct[0].Gy_ini[7,71] = -0.0280416326518497
        struct[0].Gy_ini[8,8] = -1
        struct[0].Gy_ini[8,48] = -0.000609371587085065
        struct[0].Gy_ini[8,49] = 0.00197756280128896
        struct[0].Gy_ini[8,50] = 0.00121874317417020
        struct[0].Gy_ini[8,51] = -0.00395512560257798
        struct[0].Gy_ini[8,52] = -0.000609371587085133
        struct[0].Gy_ini[8,53] = 0.00197756280128902
        struct[0].Gy_ini[8,54] = -0.000609367353078228
        struct[0].Gy_ini[8,55] = 0.00197756107030516
        struct[0].Gy_ini[8,56] = 0.00312313470615652
        struct[0].Gy_ini[8,57] = -0.0153815221406104
        struct[0].Gy_ini[8,58] = -0.000609367353078295
        struct[0].Gy_ini[8,59] = 0.00197756107030522
        struct[0].Gy_ini[8,60] = -0.000583962651075503
        struct[0].Gy_ini[8,61] = 0.00195672650234414
        struct[0].Gy_ini[8,62] = 0.00116792530215107
        struct[0].Gy_ini[8,63] = -0.00391345300468834
        struct[0].Gy_ini[8,64] = -0.000583962651075570
        struct[0].Gy_ini[8,65] = 0.00195672650234420
        struct[0].Gy_ini[8,66] = -0.0280416326518423
        struct[0].Gy_ini[8,67] = 0.0971900621093946
        struct[0].Gy_ini[8,68] = 0.0280416326518527
        struct[0].Gy_ini[8,69] = -0.0971900621093876
        struct[0].Gy_ini[9,9] = -1
        struct[0].Gy_ini[9,48] = -0.00197756280128896
        struct[0].Gy_ini[9,49] = -0.000609371587085065
        struct[0].Gy_ini[9,50] = 0.00395512560257798
        struct[0].Gy_ini[9,51] = 0.00121874317417020
        struct[0].Gy_ini[9,52] = -0.00197756280128902
        struct[0].Gy_ini[9,53] = -0.000609371587085133
        struct[0].Gy_ini[9,54] = -0.00197756107030516
        struct[0].Gy_ini[9,55] = -0.000609367353078228
        struct[0].Gy_ini[9,56] = 0.0153815221406104
        struct[0].Gy_ini[9,57] = 0.00312313470615652
        struct[0].Gy_ini[9,58] = -0.00197756107030522
        struct[0].Gy_ini[9,59] = -0.000609367353078295
        struct[0].Gy_ini[9,60] = -0.00195672650234414
        struct[0].Gy_ini[9,61] = -0.000583962651075503
        struct[0].Gy_ini[9,62] = 0.00391345300468834
        struct[0].Gy_ini[9,63] = 0.00116792530215107
        struct[0].Gy_ini[9,64] = -0.00195672650234420
        struct[0].Gy_ini[9,65] = -0.000583962651075570
        struct[0].Gy_ini[9,66] = -0.0971900621093946
        struct[0].Gy_ini[9,67] = -0.0280416326518423
        struct[0].Gy_ini[9,68] = 0.0971900621093876
        struct[0].Gy_ini[9,69] = 0.0280416326518527
        struct[0].Gy_ini[10,10] = -1
        struct[0].Gy_ini[10,48] = -0.000609371587085115
        struct[0].Gy_ini[10,49] = 0.00197756280128897
        struct[0].Gy_ini[10,50] = -0.000609371587085118
        struct[0].Gy_ini[10,51] = 0.00197756280128904
        struct[0].Gy_ini[10,52] = 0.00121874317417023
        struct[0].Gy_ini[10,53] = -0.00395512560257801
        struct[0].Gy_ini[10,54] = -0.000609367353078276
        struct[0].Gy_ini[10,55] = 0.00197756107030517
        struct[0].Gy_ini[10,56] = -0.000609367353078280
        struct[0].Gy_ini[10,57] = 0.00197756107030524
        struct[0].Gy_ini[10,58] = 0.00312313470615656
        struct[0].Gy_ini[10,59] = -0.0153815221406104
        struct[0].Gy_ini[10,60] = -0.000583962651075551
        struct[0].Gy_ini[10,61] = 0.00195672650234415
        struct[0].Gy_ini[10,62] = -0.000583962651075555
        struct[0].Gy_ini[10,63] = 0.00195672650234422
        struct[0].Gy_ini[10,64] = 0.00116792530215111
        struct[0].Gy_ini[10,65] = -0.00391345300468836
        struct[0].Gy_ini[10,68] = -0.0280416326518508
        struct[0].Gy_ini[10,69] = 0.0971900621093912
        struct[0].Gy_ini[10,70] = 0.0280416326518462
        struct[0].Gy_ini[10,71] = -0.0971900621093924
        struct[0].Gy_ini[11,11] = -1
        struct[0].Gy_ini[11,48] = -0.00197756280128897
        struct[0].Gy_ini[11,49] = -0.000609371587085115
        struct[0].Gy_ini[11,50] = -0.00197756280128904
        struct[0].Gy_ini[11,51] = -0.000609371587085118
        struct[0].Gy_ini[11,52] = 0.00395512560257801
        struct[0].Gy_ini[11,53] = 0.00121874317417023
        struct[0].Gy_ini[11,54] = -0.00197756107030517
        struct[0].Gy_ini[11,55] = -0.000609367353078276
        struct[0].Gy_ini[11,56] = -0.00197756107030524
        struct[0].Gy_ini[11,57] = -0.000609367353078280
        struct[0].Gy_ini[11,58] = 0.0153815221406104
        struct[0].Gy_ini[11,59] = 0.00312313470615656
        struct[0].Gy_ini[11,60] = -0.00195672650234415
        struct[0].Gy_ini[11,61] = -0.000583962651075551
        struct[0].Gy_ini[11,62] = -0.00195672650234422
        struct[0].Gy_ini[11,63] = -0.000583962651075555
        struct[0].Gy_ini[11,64] = 0.00391345300468836
        struct[0].Gy_ini[11,65] = 0.00116792530215111
        struct[0].Gy_ini[11,68] = -0.0971900621093912
        struct[0].Gy_ini[11,69] = -0.0280416326518508
        struct[0].Gy_ini[11,70] = 0.0971900621093924
        struct[0].Gy_ini[11,71] = 0.0280416326518462
        struct[0].Gy_ini[12,12] = -1
        struct[0].Gy_ini[12,48] = 0.00116793362771941
        struct[0].Gy_ini[12,49] = -0.00391345649507333
        struct[0].Gy_ini[12,50] = -0.000583966813859694
        struct[0].Gy_ini[12,51] = 0.00195672824753664
        struct[0].Gy_ini[12,52] = -0.000583966813859713
        struct[0].Gy_ini[12,53] = 0.00195672824753669
        struct[0].Gy_ini[12,54] = 0.00116792530215105
        struct[0].Gy_ini[12,55] = -0.00391345300468828
        struct[0].Gy_ini[12,56] = -0.000583962651075518
        struct[0].Gy_ini[12,57] = 0.00195672650234412
        struct[0].Gy_ini[12,58] = -0.000583962651075537
        struct[0].Gy_ini[12,59] = 0.00195672650234417
        struct[0].Gy_ini[12,60] = 0.00307230032548127
        struct[0].Gy_ini[12,61] = -0.0153398425335145
        struct[0].Gy_ini[12,62] = -0.000583950162740626
        struct[0].Gy_ini[12,63] = 0.00195672126675721
        struct[0].Gy_ini[12,64] = -0.000583950162740645
        struct[0].Gy_ini[12,65] = 0.00195672126675726
        struct[0].Gy_ini[12,66] = 0.0280410164125591
        struct[0].Gy_ini[12,67] = -0.0971897971186317
        struct[0].Gy_ini[12,70] = -0.0280410164125642
        struct[0].Gy_ini[12,71] = 0.0971897971186269
        struct[0].Gy_ini[13,13] = -1
        struct[0].Gy_ini[13,48] = 0.00391345649507333
        struct[0].Gy_ini[13,49] = 0.00116793362771941
        struct[0].Gy_ini[13,50] = -0.00195672824753664
        struct[0].Gy_ini[13,51] = -0.000583966813859694
        struct[0].Gy_ini[13,52] = -0.00195672824753669
        struct[0].Gy_ini[13,53] = -0.000583966813859713
        struct[0].Gy_ini[13,54] = 0.00391345300468828
        struct[0].Gy_ini[13,55] = 0.00116792530215105
        struct[0].Gy_ini[13,56] = -0.00195672650234412
        struct[0].Gy_ini[13,57] = -0.000583962651075518
        struct[0].Gy_ini[13,58] = -0.00195672650234417
        struct[0].Gy_ini[13,59] = -0.000583962651075537
        struct[0].Gy_ini[13,60] = 0.0153398425335145
        struct[0].Gy_ini[13,61] = 0.00307230032548127
        struct[0].Gy_ini[13,62] = -0.00195672126675721
        struct[0].Gy_ini[13,63] = -0.000583950162740626
        struct[0].Gy_ini[13,64] = -0.00195672126675726
        struct[0].Gy_ini[13,65] = -0.000583950162740645
        struct[0].Gy_ini[13,66] = 0.0971897971186317
        struct[0].Gy_ini[13,67] = 0.0280410164125591
        struct[0].Gy_ini[13,70] = -0.0971897971186269
        struct[0].Gy_ini[13,71] = -0.0280410164125642
        struct[0].Gy_ini[14,14] = -1
        struct[0].Gy_ini[14,48] = -0.000583966813859680
        struct[0].Gy_ini[14,49] = 0.00195672824753666
        struct[0].Gy_ini[14,50] = 0.00116793362771943
        struct[0].Gy_ini[14,51] = -0.00391345649507338
        struct[0].Gy_ini[14,52] = -0.000583966813859745
        struct[0].Gy_ini[14,53] = 0.00195672824753672
        struct[0].Gy_ini[14,54] = -0.000583962651075503
        struct[0].Gy_ini[14,55] = 0.00195672650234414
        struct[0].Gy_ini[14,56] = 0.00116792530215107
        struct[0].Gy_ini[14,57] = -0.00391345300468834
        struct[0].Gy_ini[14,58] = -0.000583962651075569
        struct[0].Gy_ini[14,59] = 0.00195672650234420
        struct[0].Gy_ini[14,60] = -0.000583950162740612
        struct[0].Gy_ini[14,61] = 0.00195672126675723
        struct[0].Gy_ini[14,62] = 0.00307230032548129
        struct[0].Gy_ini[14,63] = -0.0153398425335145
        struct[0].Gy_ini[14,64] = -0.000583950162740677
        struct[0].Gy_ini[14,65] = 0.00195672126675729
        struct[0].Gy_ini[14,66] = -0.0280410164125570
        struct[0].Gy_ini[14,67] = 0.0971897971186340
        struct[0].Gy_ini[14,68] = 0.0280410164125671
        struct[0].Gy_ini[14,69] = -0.0971897971186271
        struct[0].Gy_ini[15,15] = -1
        struct[0].Gy_ini[15,48] = -0.00195672824753666
        struct[0].Gy_ini[15,49] = -0.000583966813859680
        struct[0].Gy_ini[15,50] = 0.00391345649507338
        struct[0].Gy_ini[15,51] = 0.00116793362771943
        struct[0].Gy_ini[15,52] = -0.00195672824753672
        struct[0].Gy_ini[15,53] = -0.000583966813859745
        struct[0].Gy_ini[15,54] = -0.00195672650234414
        struct[0].Gy_ini[15,55] = -0.000583962651075503
        struct[0].Gy_ini[15,56] = 0.00391345300468834
        struct[0].Gy_ini[15,57] = 0.00116792530215107
        struct[0].Gy_ini[15,58] = -0.00195672650234420
        struct[0].Gy_ini[15,59] = -0.000583962651075569
        struct[0].Gy_ini[15,60] = -0.00195672126675723
        struct[0].Gy_ini[15,61] = -0.000583950162740612
        struct[0].Gy_ini[15,62] = 0.0153398425335145
        struct[0].Gy_ini[15,63] = 0.00307230032548129
        struct[0].Gy_ini[15,64] = -0.00195672126675729
        struct[0].Gy_ini[15,65] = -0.000583950162740677
        struct[0].Gy_ini[15,66] = -0.0971897971186340
        struct[0].Gy_ini[15,67] = -0.0280410164125570
        struct[0].Gy_ini[15,68] = 0.0971897971186271
        struct[0].Gy_ini[15,69] = 0.0280410164125671
        struct[0].Gy_ini[16,16] = -1
        struct[0].Gy_ini[16,48] = -0.000583966813859728
        struct[0].Gy_ini[16,49] = 0.00195672824753667
        struct[0].Gy_ini[16,50] = -0.000583966813859731
        struct[0].Gy_ini[16,51] = 0.00195672824753674
        struct[0].Gy_ini[16,52] = 0.00116793362771946
        struct[0].Gy_ini[16,53] = -0.00391345649507341
        struct[0].Gy_ini[16,54] = -0.000583962651075551
        struct[0].Gy_ini[16,55] = 0.00195672650234415
        struct[0].Gy_ini[16,56] = -0.000583962651075554
        struct[0].Gy_ini[16,57] = 0.00195672650234422
        struct[0].Gy_ini[16,58] = 0.00116792530215111
        struct[0].Gy_ini[16,59] = -0.00391345300468836
        struct[0].Gy_ini[16,60] = -0.000583950162740659
        struct[0].Gy_ini[16,61] = 0.00195672126675724
        struct[0].Gy_ini[16,62] = -0.000583950162740663
        struct[0].Gy_ini[16,63] = 0.00195672126675731
        struct[0].Gy_ini[16,64] = 0.00307230032548132
        struct[0].Gy_ini[16,65] = -0.0153398425335145
        struct[0].Gy_ini[16,68] = -0.0280410164125652
        struct[0].Gy_ini[16,69] = 0.0971897971186306
        struct[0].Gy_ini[16,70] = 0.0280410164125607
        struct[0].Gy_ini[16,71] = -0.0971897971186319
        struct[0].Gy_ini[17,17] = -1
        struct[0].Gy_ini[17,48] = -0.00195672824753667
        struct[0].Gy_ini[17,49] = -0.000583966813859728
        struct[0].Gy_ini[17,50] = -0.00195672824753674
        struct[0].Gy_ini[17,51] = -0.000583966813859731
        struct[0].Gy_ini[17,52] = 0.00391345649507341
        struct[0].Gy_ini[17,53] = 0.00116793362771946
        struct[0].Gy_ini[17,54] = -0.00195672650234415
        struct[0].Gy_ini[17,55] = -0.000583962651075551
        struct[0].Gy_ini[17,56] = -0.00195672650234422
        struct[0].Gy_ini[17,57] = -0.000583962651075554
        struct[0].Gy_ini[17,58] = 0.00391345300468836
        struct[0].Gy_ini[17,59] = 0.00116792530215111
        struct[0].Gy_ini[17,60] = -0.00195672126675724
        struct[0].Gy_ini[17,61] = -0.000583950162740659
        struct[0].Gy_ini[17,62] = -0.00195672126675731
        struct[0].Gy_ini[17,63] = -0.000583950162740663
        struct[0].Gy_ini[17,64] = 0.0153398425335145
        struct[0].Gy_ini[17,65] = 0.00307230032548132
        struct[0].Gy_ini[17,68] = -0.0971897971186306
        struct[0].Gy_ini[17,69] = -0.0280410164125652
        struct[0].Gy_ini[17,70] = 0.0971897971186319
        struct[0].Gy_ini[17,71] = 0.0280410164125607
        struct[0].Gy_ini[18,18] = -1
        struct[0].Gy_ini[18,48] = 0.0280418380652350
        struct[0].Gy_ini[18,49] = -0.0971901504394881
        struct[0].Gy_ini[18,50] = -0.0280418380652335
        struct[0].Gy_ini[18,51] = 0.0971901504394933
        struct[0].Gy_ini[18,54] = 0.0280416326518473
        struct[0].Gy_ini[18,55] = -0.0971900621093861
        struct[0].Gy_ini[18,56] = -0.0280416326518457
        struct[0].Gy_ini[18,57] = 0.0971900621093912
        struct[0].Gy_ini[18,60] = 0.0280410164125620
        struct[0].Gy_ini[18,61] = -0.0971897971186256
        struct[0].Gy_ini[18,62] = -0.0280410164125604
        struct[0].Gy_ini[18,63] = 0.0971897971186307
        struct[0].Gy_ini[18,66] = 11.6022742434667
        struct[0].Gy_ini[18,67] = -3.08885814101954
        struct[0].Gy_ini[18,68] = 10.1945442087447
        struct[0].Gy_ini[18,69] = 1.79047234076949
        struct[0].Gy_ini[18,70] = 10.1945442087447
        struct[0].Gy_ini[18,71] = 1.79047234076923
        struct[0].Gy_ini[19,19] = -1
        struct[0].Gy_ini[19,48] = 0.0971901504394881
        struct[0].Gy_ini[19,49] = 0.0280418380652350
        struct[0].Gy_ini[19,50] = -0.0971901504394933
        struct[0].Gy_ini[19,51] = -0.0280418380652335
        struct[0].Gy_ini[19,54] = 0.0971900621093861
        struct[0].Gy_ini[19,55] = 0.0280416326518473
        struct[0].Gy_ini[19,56] = -0.0971900621093912
        struct[0].Gy_ini[19,57] = -0.0280416326518457
        struct[0].Gy_ini[19,60] = 0.0971897971186256
        struct[0].Gy_ini[19,61] = 0.0280410164125620
        struct[0].Gy_ini[19,62] = -0.0971897971186307
        struct[0].Gy_ini[19,63] = -0.0280410164125604
        struct[0].Gy_ini[19,66] = 3.08885814101954
        struct[0].Gy_ini[19,67] = 11.6022742434667
        struct[0].Gy_ini[19,68] = -1.79047234076949
        struct[0].Gy_ini[19,69] = 10.1945442087447
        struct[0].Gy_ini[19,70] = -1.79047234076923
        struct[0].Gy_ini[19,71] = 10.1945442087447
        struct[0].Gy_ini[20,20] = -1
        struct[0].Gy_ini[20,50] = 0.0280418380652370
        struct[0].Gy_ini[20,51] = -0.0971901504394930
        struct[0].Gy_ini[20,52] = -0.0280418380652384
        struct[0].Gy_ini[20,53] = 0.0971901504394895
        struct[0].Gy_ini[20,56] = 0.0280416326518492
        struct[0].Gy_ini[20,57] = -0.0971900621093910
        struct[0].Gy_ini[20,58] = -0.0280416326518506
        struct[0].Gy_ini[20,59] = 0.0971900621093875
        struct[0].Gy_ini[20,62] = 0.0280410164125637
        struct[0].Gy_ini[20,63] = -0.0971897971186304
        struct[0].Gy_ini[20,64] = -0.0280410164125652
        struct[0].Gy_ini[20,65] = 0.0971897971186269
        struct[0].Gy_ini[20,66] = 10.1945442087449
        struct[0].Gy_ini[20,67] = 1.79047234076965
        struct[0].Gy_ini[20,68] = 11.6022742434671
        struct[0].Gy_ini[20,69] = -3.08885814101935
        struct[0].Gy_ini[20,70] = 10.1945442087448
        struct[0].Gy_ini[20,71] = 1.79047234076948
        struct[0].Gy_ini[21,21] = -1
        struct[0].Gy_ini[21,50] = 0.0971901504394930
        struct[0].Gy_ini[21,51] = 0.0280418380652370
        struct[0].Gy_ini[21,52] = -0.0971901504394895
        struct[0].Gy_ini[21,53] = -0.0280418380652384
        struct[0].Gy_ini[21,56] = 0.0971900621093910
        struct[0].Gy_ini[21,57] = 0.0280416326518492
        struct[0].Gy_ini[21,58] = -0.0971900621093875
        struct[0].Gy_ini[21,59] = -0.0280416326518506
        struct[0].Gy_ini[21,62] = 0.0971897971186304
        struct[0].Gy_ini[21,63] = 0.0280410164125637
        struct[0].Gy_ini[21,64] = -0.0971897971186269
        struct[0].Gy_ini[21,65] = -0.0280410164125652
        struct[0].Gy_ini[21,66] = -1.79047234076965
        struct[0].Gy_ini[21,67] = 10.1945442087449
        struct[0].Gy_ini[21,68] = 3.08885814101935
        struct[0].Gy_ini[21,69] = 11.6022742434671
        struct[0].Gy_ini[21,70] = -1.79047234076948
        struct[0].Gy_ini[21,71] = 10.1945442087448
        struct[0].Gy_ini[22,22] = -1
        struct[0].Gy_ini[22,48] = -0.0280418380652345
        struct[0].Gy_ini[22,49] = 0.0971901504394956
        struct[0].Gy_ini[22,52] = 0.0280418380652337
        struct[0].Gy_ini[22,53] = -0.0971901504394982
        struct[0].Gy_ini[22,54] = -0.0280416326518468
        struct[0].Gy_ini[22,55] = 0.0971900621093936
        struct[0].Gy_ini[22,58] = 0.0280416326518460
        struct[0].Gy_ini[22,59] = -0.0971900621093962
        struct[0].Gy_ini[22,60] = -0.0280410164125613
        struct[0].Gy_ini[22,61] = 0.0971897971186331
        struct[0].Gy_ini[22,64] = 0.0280410164125606
        struct[0].Gy_ini[22,65] = -0.0971897971186357
        struct[0].Gy_ini[22,66] = 10.1945442087448
        struct[0].Gy_ini[22,67] = 1.79047234076953
        struct[0].Gy_ini[22,68] = 10.1945442087448
        struct[0].Gy_ini[22,69] = 1.79047234076966
        struct[0].Gy_ini[22,70] = 11.6022742434669
        struct[0].Gy_ini[22,71] = -3.08885814101960
        struct[0].Gy_ini[23,23] = -1
        struct[0].Gy_ini[23,48] = -0.0971901504394956
        struct[0].Gy_ini[23,49] = -0.0280418380652345
        struct[0].Gy_ini[23,52] = 0.0971901504394982
        struct[0].Gy_ini[23,53] = 0.0280418380652337
        struct[0].Gy_ini[23,54] = -0.0971900621093936
        struct[0].Gy_ini[23,55] = -0.0280416326518468
        struct[0].Gy_ini[23,58] = 0.0971900621093962
        struct[0].Gy_ini[23,59] = 0.0280416326518460
        struct[0].Gy_ini[23,60] = -0.0971897971186331
        struct[0].Gy_ini[23,61] = -0.0280410164125613
        struct[0].Gy_ini[23,64] = 0.0971897971186357
        struct[0].Gy_ini[23,65] = 0.0280410164125606
        struct[0].Gy_ini[23,66] = -1.79047234076953
        struct[0].Gy_ini[23,67] = 10.1945442087448
        struct[0].Gy_ini[23,68] = -1.79047234076966
        struct[0].Gy_ini[23,69] = 10.1945442087448
        struct[0].Gy_ini[23,70] = 3.08885814101960
        struct[0].Gy_ini[23,71] = 11.6022742434669
        struct[0].Gy_ini[24,24] = -1
        struct[0].Gy_ini[24,48] = 0.0764099708538850
        struct[0].Gy_ini[24,49] = -0.127279074345343
        struct[0].Gy_ini[24,50] = -0.0382049854269419
        struct[0].Gy_ini[24,51] = 0.0636395371726706
        struct[0].Gy_ini[24,52] = -0.0382049854269430
        struct[0].Gy_ini[24,53] = 0.0636395371726721
        struct[0].Gy_ini[24,54] = 0.0764096462087186
        struct[0].Gy_ini[24,55] = -0.127279026494919
        struct[0].Gy_ini[24,56] = -0.0382048231043588
        struct[0].Gy_ini[24,57] = 0.0636395132474590
        struct[0].Gy_ini[24,58] = -0.0382048231043599
        struct[0].Gy_ini[24,59] = 0.0636395132474605
        struct[0].Gy_ini[24,60] = 0.0764086722742935
        struct[0].Gy_ini[24,61] = -0.127278882942674
        struct[0].Gy_ini[24,62] = -0.0382043361371462
        struct[0].Gy_ini[24,63] = 0.0636394414713363
        struct[0].Gy_ini[24,64] = -0.0382043361371473
        struct[0].Gy_ini[24,65] = 0.0636394414713379
        struct[0].Gy_ini[24,66] = 1.91798392779186
        struct[0].Gy_ini[24,67] = -3.19497213887046
        struct[0].Gy_ini[24,70] = -1.91798392779199
        struct[0].Gy_ini[24,71] = 3.19497213887024
        struct[0].Gy_ini[25,25] = -1
        struct[0].Gy_ini[25,48] = 0.127279074345343
        struct[0].Gy_ini[25,49] = 0.0764099708538850
        struct[0].Gy_ini[25,50] = -0.0636395371726706
        struct[0].Gy_ini[25,51] = -0.0382049854269419
        struct[0].Gy_ini[25,52] = -0.0636395371726721
        struct[0].Gy_ini[25,53] = -0.0382049854269430
        struct[0].Gy_ini[25,54] = 0.127279026494919
        struct[0].Gy_ini[25,55] = 0.0764096462087186
        struct[0].Gy_ini[25,56] = -0.0636395132474590
        struct[0].Gy_ini[25,57] = -0.0382048231043588
        struct[0].Gy_ini[25,58] = -0.0636395132474605
        struct[0].Gy_ini[25,59] = -0.0382048231043599
        struct[0].Gy_ini[25,60] = 0.127278882942674
        struct[0].Gy_ini[25,61] = 0.0764086722742935
        struct[0].Gy_ini[25,62] = -0.0636394414713363
        struct[0].Gy_ini[25,63] = -0.0382043361371462
        struct[0].Gy_ini[25,64] = -0.0636394414713379
        struct[0].Gy_ini[25,65] = -0.0382043361371473
        struct[0].Gy_ini[25,66] = 3.19497213887046
        struct[0].Gy_ini[25,67] = 1.91798392779186
        struct[0].Gy_ini[25,70] = -3.19497213887024
        struct[0].Gy_ini[25,71] = -1.91798392779199
        struct[0].Gy_ini[26,26] = -1
        struct[0].Gy_ini[26,48] = -0.0382049854269416
        struct[0].Gy_ini[26,49] = 0.0636395371726714
        struct[0].Gy_ini[26,50] = 0.0764099708538861
        struct[0].Gy_ini[26,51] = -0.127279074345344
        struct[0].Gy_ini[26,52] = -0.0382049854269445
        struct[0].Gy_ini[26,53] = 0.0636395371726729
        struct[0].Gy_ini[26,54] = -0.0382048231043584
        struct[0].Gy_ini[26,55] = 0.0636395132474598
        struct[0].Gy_ini[26,56] = 0.0764096462087197
        struct[0].Gy_ini[26,57] = -0.127279026494921
        struct[0].Gy_ini[26,58] = -0.0382048231043613
        struct[0].Gy_ini[26,59] = 0.0636395132474612
        struct[0].Gy_ini[26,60] = -0.0382043361371459
        struct[0].Gy_ini[26,61] = 0.0636394414713371
        struct[0].Gy_ini[26,62] = 0.0764086722742946
        struct[0].Gy_ini[26,63] = -0.127278882942676
        struct[0].Gy_ini[26,64] = -0.0382043361371488
        struct[0].Gy_ini[26,65] = 0.0636394414713386
        struct[0].Gy_ini[26,66] = -1.91798392779181
        struct[0].Gy_ini[26,67] = 3.19497213887056
        struct[0].Gy_ini[26,68] = 1.91798392779210
        struct[0].Gy_ini[26,69] = -3.19497213887022
        struct[0].Gy_ini[27,27] = -1
        struct[0].Gy_ini[27,48] = -0.0636395371726714
        struct[0].Gy_ini[27,49] = -0.0382049854269416
        struct[0].Gy_ini[27,50] = 0.127279074345344
        struct[0].Gy_ini[27,51] = 0.0764099708538861
        struct[0].Gy_ini[27,52] = -0.0636395371726729
        struct[0].Gy_ini[27,53] = -0.0382049854269445
        struct[0].Gy_ini[27,54] = -0.0636395132474598
        struct[0].Gy_ini[27,55] = -0.0382048231043584
        struct[0].Gy_ini[27,56] = 0.127279026494921
        struct[0].Gy_ini[27,57] = 0.0764096462087197
        struct[0].Gy_ini[27,58] = -0.0636395132474612
        struct[0].Gy_ini[27,59] = -0.0382048231043613
        struct[0].Gy_ini[27,60] = -0.0636394414713371
        struct[0].Gy_ini[27,61] = -0.0382043361371459
        struct[0].Gy_ini[27,62] = 0.127278882942676
        struct[0].Gy_ini[27,63] = 0.0764086722742946
        struct[0].Gy_ini[27,64] = -0.0636394414713386
        struct[0].Gy_ini[27,65] = -0.0382043361371488
        struct[0].Gy_ini[27,66] = -3.19497213887056
        struct[0].Gy_ini[27,67] = -1.91798392779181
        struct[0].Gy_ini[27,68] = 3.19497213887022
        struct[0].Gy_ini[27,69] = 1.91798392779210
        struct[0].Gy_ini[28,28] = -1
        struct[0].Gy_ini[28,48] = -0.0382049854269434
        struct[0].Gy_ini[28,49] = 0.0636395371726713
        struct[0].Gy_ini[28,50] = -0.0382049854269442
        struct[0].Gy_ini[28,51] = 0.0636395371726737
        struct[0].Gy_ini[28,52] = 0.0764099708538875
        struct[0].Gy_ini[28,53] = -0.127279074345345
        struct[0].Gy_ini[28,54] = -0.0382048231043602
        struct[0].Gy_ini[28,55] = 0.0636395132474596
        struct[0].Gy_ini[28,56] = -0.0382048231043610
        struct[0].Gy_ini[28,57] = 0.0636395132474621
        struct[0].Gy_ini[28,58] = 0.0764096462087212
        struct[0].Gy_ini[28,59] = -0.127279026494922
        struct[0].Gy_ini[28,60] = -0.0382043361371476
        struct[0].Gy_ini[28,61] = 0.0636394414713370
        struct[0].Gy_ini[28,62] = -0.0382043361371484
        struct[0].Gy_ini[28,63] = 0.0636394414713394
        struct[0].Gy_ini[28,64] = 0.0764086722742961
        struct[0].Gy_ini[28,65] = -0.127278882942676
        struct[0].Gy_ini[28,68] = -1.91798392779206
        struct[0].Gy_ini[28,69] = 3.19497213887036
        struct[0].Gy_ini[28,70] = 1.91798392779192
        struct[0].Gy_ini[28,71] = -3.19497213887045
        struct[0].Gy_ini[29,29] = -1
        struct[0].Gy_ini[29,48] = -0.0636395371726713
        struct[0].Gy_ini[29,49] = -0.0382049854269434
        struct[0].Gy_ini[29,50] = -0.0636395371726737
        struct[0].Gy_ini[29,51] = -0.0382049854269442
        struct[0].Gy_ini[29,52] = 0.127279074345345
        struct[0].Gy_ini[29,53] = 0.0764099708538875
        struct[0].Gy_ini[29,54] = -0.0636395132474596
        struct[0].Gy_ini[29,55] = -0.0382048231043602
        struct[0].Gy_ini[29,56] = -0.0636395132474621
        struct[0].Gy_ini[29,57] = -0.0382048231043610
        struct[0].Gy_ini[29,58] = 0.127279026494922
        struct[0].Gy_ini[29,59] = 0.0764096462087212
        struct[0].Gy_ini[29,60] = -0.0636394414713370
        struct[0].Gy_ini[29,61] = -0.0382043361371476
        struct[0].Gy_ini[29,62] = -0.0636394414713394
        struct[0].Gy_ini[29,63] = -0.0382043361371484
        struct[0].Gy_ini[29,64] = 0.127278882942676
        struct[0].Gy_ini[29,65] = 0.0764086722742961
        struct[0].Gy_ini[29,68] = -3.19497213887036
        struct[0].Gy_ini[29,69] = -1.91798392779206
        struct[0].Gy_ini[29,70] = 3.19497213887045
        struct[0].Gy_ini[29,71] = 1.91798392779192
        struct[0].Gy_ini[30,30] = -1
        struct[0].Gy_ini[30,48] = 0.0318681229122168
        struct[0].Gy_ini[30,49] = -0.100328108879386
        struct[0].Gy_ini[30,50] = -0.0318681229122152
        struct[0].Gy_ini[30,51] = 0.100328108879391
        struct[0].Gy_ini[30,54] = 0.0305931173773991
        struct[0].Gy_ini[30,55] = -0.0992822970142200
        struct[0].Gy_ini[30,56] = -0.0305931173773975
        struct[0].Gy_ini[30,57] = 0.0992822970142252
        struct[0].Gy_ini[30,60] = 0.0293176867112793
        struct[0].Gy_ini[30,61] = -0.0982363113431474
        struct[0].Gy_ini[30,62] = -0.0293176867112777
        struct[0].Gy_ini[30,63] = 0.0982363113431526
        struct[0].Gy_ini[30,66] = 11.6035242966407
        struct[0].Gy_ini[30,67] = -3.08738963687029
        struct[0].Gy_ini[30,68] = 10.1957014483333
        struct[0].Gy_ini[30,69] = 1.79198075607073
        struct[0].Gy_ini[30,70] = 10.1957014483333
        struct[0].Gy_ini[30,71] = 1.79198075607047
        struct[0].Gy_ini[31,31] = -1
        struct[0].Gy_ini[31,48] = 0.100328108879386
        struct[0].Gy_ini[31,49] = 0.0318681229122168
        struct[0].Gy_ini[31,50] = -0.100328108879391
        struct[0].Gy_ini[31,51] = -0.0318681229122152
        struct[0].Gy_ini[31,54] = 0.0992822970142200
        struct[0].Gy_ini[31,55] = 0.0305931173773991
        struct[0].Gy_ini[31,56] = -0.0992822970142252
        struct[0].Gy_ini[31,57] = -0.0305931173773975
        struct[0].Gy_ini[31,60] = 0.0982363113431474
        struct[0].Gy_ini[31,61] = 0.0293176867112793
        struct[0].Gy_ini[31,62] = -0.0982363113431526
        struct[0].Gy_ini[31,63] = -0.0293176867112777
        struct[0].Gy_ini[31,66] = 3.08738963687029
        struct[0].Gy_ini[31,67] = 11.6035242966407
        struct[0].Gy_ini[31,68] = -1.79198075607073
        struct[0].Gy_ini[31,69] = 10.1957014483333
        struct[0].Gy_ini[31,70] = -1.79198075607047
        struct[0].Gy_ini[31,71] = 10.1957014483333
        struct[0].Gy_ini[32,32] = -1
        struct[0].Gy_ini[32,50] = 0.0318681229122188
        struct[0].Gy_ini[32,51] = -0.100328108879391
        struct[0].Gy_ini[32,52] = -0.0318681229122203
        struct[0].Gy_ini[32,53] = 0.100328108879387
        struct[0].Gy_ini[32,56] = 0.0305931173774011
        struct[0].Gy_ini[32,57] = -0.0992822970142250
        struct[0].Gy_ini[32,58] = -0.0305931173774025
        struct[0].Gy_ini[32,59] = 0.0992822970142214
        struct[0].Gy_ini[32,62] = 0.0293176867112812
        struct[0].Gy_ini[32,63] = -0.0982363113431523
        struct[0].Gy_ini[32,64] = -0.0293176867112828
        struct[0].Gy_ini[32,65] = 0.0982363113431489
        struct[0].Gy_ini[32,66] = 10.1957014483335
        struct[0].Gy_ini[32,67] = 1.79198075607090
        struct[0].Gy_ini[32,68] = 11.6035242966410
        struct[0].Gy_ini[32,69] = -3.08738963687010
        struct[0].Gy_ini[32,70] = 10.1957014483334
        struct[0].Gy_ini[32,71] = 1.79198075607072
        struct[0].Gy_ini[33,33] = -1
        struct[0].Gy_ini[33,50] = 0.100328108879391
        struct[0].Gy_ini[33,51] = 0.0318681229122188
        struct[0].Gy_ini[33,52] = -0.100328108879387
        struct[0].Gy_ini[33,53] = -0.0318681229122203
        struct[0].Gy_ini[33,56] = 0.0992822970142250
        struct[0].Gy_ini[33,57] = 0.0305931173774011
        struct[0].Gy_ini[33,58] = -0.0992822970142214
        struct[0].Gy_ini[33,59] = -0.0305931173774025
        struct[0].Gy_ini[33,62] = 0.0982363113431523
        struct[0].Gy_ini[33,63] = 0.0293176867112812
        struct[0].Gy_ini[33,64] = -0.0982363113431489
        struct[0].Gy_ini[33,65] = -0.0293176867112828
        struct[0].Gy_ini[33,66] = -1.79198075607090
        struct[0].Gy_ini[33,67] = 10.1957014483335
        struct[0].Gy_ini[33,68] = 3.08738963687010
        struct[0].Gy_ini[33,69] = 11.6035242966410
        struct[0].Gy_ini[33,70] = -1.79198075607072
        struct[0].Gy_ini[33,71] = 10.1957014483334
        struct[0].Gy_ini[34,34] = -1
        struct[0].Gy_ini[34,48] = -0.0318681229122163
        struct[0].Gy_ini[34,49] = 0.100328108879394
        struct[0].Gy_ini[34,52] = 0.0318681229122155
        struct[0].Gy_ini[34,53] = -0.100328108879396
        struct[0].Gy_ini[34,54] = -0.0305931173773985
        struct[0].Gy_ini[34,55] = 0.0992822970142275
        struct[0].Gy_ini[34,58] = 0.0305931173773979
        struct[0].Gy_ini[34,59] = -0.0992822970142302
        struct[0].Gy_ini[34,60] = -0.0293176867112786
        struct[0].Gy_ini[34,61] = 0.0982363113431550
        struct[0].Gy_ini[34,64] = 0.0293176867112779
        struct[0].Gy_ini[34,65] = -0.0982363113431575
        struct[0].Gy_ini[34,66] = 10.1957014483334
        struct[0].Gy_ini[34,67] = 1.79198075607079
        struct[0].Gy_ini[34,68] = 10.1957014483334
        struct[0].Gy_ini[34,69] = 1.79198075607092
        struct[0].Gy_ini[34,70] = 11.6035242966408
        struct[0].Gy_ini[34,71] = -3.08738963687035
        struct[0].Gy_ini[35,35] = -1
        struct[0].Gy_ini[35,48] = -0.100328108879394
        struct[0].Gy_ini[35,49] = -0.0318681229122163
        struct[0].Gy_ini[35,52] = 0.100328108879396
        struct[0].Gy_ini[35,53] = 0.0318681229122155
        struct[0].Gy_ini[35,54] = -0.0992822970142275
        struct[0].Gy_ini[35,55] = -0.0305931173773985
        struct[0].Gy_ini[35,58] = 0.0992822970142302
        struct[0].Gy_ini[35,59] = 0.0305931173773979
        struct[0].Gy_ini[35,60] = -0.0982363113431550
        struct[0].Gy_ini[35,61] = -0.0293176867112786
        struct[0].Gy_ini[35,64] = 0.0982363113431575
        struct[0].Gy_ini[35,65] = 0.0293176867112779
        struct[0].Gy_ini[35,66] = -1.79198075607079
        struct[0].Gy_ini[35,67] = 10.1957014483334
        struct[0].Gy_ini[35,68] = -1.79198075607092
        struct[0].Gy_ini[35,69] = 10.1957014483334
        struct[0].Gy_ini[35,70] = 3.08738963687035
        struct[0].Gy_ini[35,71] = 11.6035242966408
        struct[0].Gy_ini[36,36] = -1
        struct[0].Gy_ini[36,48] = 0.0305931173773991
        struct[0].Gy_ini[36,49] = -0.0992822970142201
        struct[0].Gy_ini[36,50] = -0.0305931173773975
        struct[0].Gy_ini[36,51] = 0.0992822970142253
        struct[0].Gy_ini[36,54] = 0.0305929048117480
        struct[0].Gy_ini[36,55] = -0.0992822101112667
        struct[0].Gy_ini[36,56] = -0.0305929048117464
        struct[0].Gy_ini[36,57] = 0.0992822101112719
        struct[0].Gy_ini[36,60] = 0.0293174777213171
        struct[0].Gy_ini[36,61] = -0.0982362237268541
        struct[0].Gy_ini[36,62] = -0.0293174777213155
        struct[0].Gy_ini[36,63] = 0.0982362237268593
        struct[0].Gy_ini[36,66] = 11.6033854030100
        struct[0].Gy_ini[36,67] = -3.08755280951041
        struct[0].Gy_ini[36,68] = 10.1955728673526
        struct[0].Gy_ini[36,69] = 1.79181314887337
        struct[0].Gy_ini[36,70] = 10.1955728673526
        struct[0].Gy_ini[36,71] = 1.79181314887312
        struct[0].Gy_ini[37,37] = -1
        struct[0].Gy_ini[37,48] = 0.0992822970142201
        struct[0].Gy_ini[37,49] = 0.0305931173773991
        struct[0].Gy_ini[37,50] = -0.0992822970142253
        struct[0].Gy_ini[37,51] = -0.0305931173773975
        struct[0].Gy_ini[37,54] = 0.0992822101112667
        struct[0].Gy_ini[37,55] = 0.0305929048117480
        struct[0].Gy_ini[37,56] = -0.0992822101112719
        struct[0].Gy_ini[37,57] = -0.0305929048117464
        struct[0].Gy_ini[37,60] = 0.0982362237268541
        struct[0].Gy_ini[37,61] = 0.0293174777213171
        struct[0].Gy_ini[37,62] = -0.0982362237268593
        struct[0].Gy_ini[37,63] = -0.0293174777213155
        struct[0].Gy_ini[37,66] = 3.08755280951041
        struct[0].Gy_ini[37,67] = 11.6033854030100
        struct[0].Gy_ini[37,68] = -1.79181314887337
        struct[0].Gy_ini[37,69] = 10.1955728673526
        struct[0].Gy_ini[37,70] = -1.79181314887312
        struct[0].Gy_ini[37,71] = 10.1955728673526
        struct[0].Gy_ini[38,38] = -1
        struct[0].Gy_ini[38,50] = 0.0305931173774011
        struct[0].Gy_ini[38,51] = -0.0992822970142250
        struct[0].Gy_ini[38,52] = -0.0305931173774025
        struct[0].Gy_ini[38,53] = 0.0992822970142215
        struct[0].Gy_ini[38,56] = 0.0305929048117499
        struct[0].Gy_ini[38,57] = -0.0992822101112716
        struct[0].Gy_ini[38,58] = -0.0305929048117514
        struct[0].Gy_ini[38,59] = 0.0992822101112681
        struct[0].Gy_ini[38,62] = 0.0293174777213190
        struct[0].Gy_ini[38,63] = -0.0982362237268590
        struct[0].Gy_ini[38,64] = -0.0293174777213205
        struct[0].Gy_ini[38,65] = 0.0982362237268555
        struct[0].Gy_ini[38,66] = 10.1955728673528
        struct[0].Gy_ini[38,67] = 1.79181314887354
        struct[0].Gy_ini[38,68] = 11.6033854030104
        struct[0].Gy_ini[38,69] = -3.08755280951023
        struct[0].Gy_ini[38,70] = 10.1955728673527
        struct[0].Gy_ini[38,71] = 1.79181314887337
        struct[0].Gy_ini[39,39] = -1
        struct[0].Gy_ini[39,50] = 0.0992822970142250
        struct[0].Gy_ini[39,51] = 0.0305931173774011
        struct[0].Gy_ini[39,52] = -0.0992822970142215
        struct[0].Gy_ini[39,53] = -0.0305931173774025
        struct[0].Gy_ini[39,56] = 0.0992822101112716
        struct[0].Gy_ini[39,57] = 0.0305929048117499
        struct[0].Gy_ini[39,58] = -0.0992822101112681
        struct[0].Gy_ini[39,59] = -0.0305929048117514
        struct[0].Gy_ini[39,62] = 0.0982362237268590
        struct[0].Gy_ini[39,63] = 0.0293174777213190
        struct[0].Gy_ini[39,64] = -0.0982362237268555
        struct[0].Gy_ini[39,65] = -0.0293174777213205
        struct[0].Gy_ini[39,66] = -1.79181314887354
        struct[0].Gy_ini[39,67] = 10.1955728673528
        struct[0].Gy_ini[39,68] = 3.08755280951023
        struct[0].Gy_ini[39,69] = 11.6033854030104
        struct[0].Gy_ini[39,70] = -1.79181314887337
        struct[0].Gy_ini[39,71] = 10.1955728673527
        struct[0].Gy_ini[40,40] = -1
        struct[0].Gy_ini[40,48] = -0.0305931173773985
        struct[0].Gy_ini[40,49] = 0.0992822970142276
        struct[0].Gy_ini[40,52] = 0.0305931173773978
        struct[0].Gy_ini[40,53] = -0.0992822970142302
        struct[0].Gy_ini[40,54] = -0.0305929048117474
        struct[0].Gy_ini[40,55] = 0.0992822101112742
        struct[0].Gy_ini[40,58] = 0.0305929048117467
        struct[0].Gy_ini[40,59] = -0.0992822101112768
        struct[0].Gy_ini[40,60] = -0.0293174777213164
        struct[0].Gy_ini[40,61] = 0.0982362237268616
        struct[0].Gy_ini[40,64] = 0.0293174777213157
        struct[0].Gy_ini[40,65] = -0.0982362237268642
        struct[0].Gy_ini[40,66] = 10.1955728673527
        struct[0].Gy_ini[40,67] = 1.79181314887342
        struct[0].Gy_ini[40,68] = 10.1955728673527
        struct[0].Gy_ini[40,69] = 1.79181314887355
        struct[0].Gy_ini[40,70] = 11.6033854030101
        struct[0].Gy_ini[40,71] = -3.08755280951048
        struct[0].Gy_ini[41,41] = -1
        struct[0].Gy_ini[41,48] = -0.0992822970142276
        struct[0].Gy_ini[41,49] = -0.0305931173773985
        struct[0].Gy_ini[41,52] = 0.0992822970142302
        struct[0].Gy_ini[41,53] = 0.0305931173773978
        struct[0].Gy_ini[41,54] = -0.0992822101112742
        struct[0].Gy_ini[41,55] = -0.0305929048117474
        struct[0].Gy_ini[41,58] = 0.0992822101112768
        struct[0].Gy_ini[41,59] = 0.0305929048117467
        struct[0].Gy_ini[41,60] = -0.0982362237268616
        struct[0].Gy_ini[41,61] = -0.0293174777213164
        struct[0].Gy_ini[41,64] = 0.0982362237268642
        struct[0].Gy_ini[41,65] = 0.0293174777213157
        struct[0].Gy_ini[41,66] = -1.79181314887342
        struct[0].Gy_ini[41,67] = 10.1955728673527
        struct[0].Gy_ini[41,68] = -1.79181314887355
        struct[0].Gy_ini[41,69] = 10.1955728673527
        struct[0].Gy_ini[41,70] = 3.08755280951048
        struct[0].Gy_ini[41,71] = 11.6033854030101
        struct[0].Gy_ini[42,42] = -1
        struct[0].Gy_ini[42,48] = 0.0293176867112792
        struct[0].Gy_ini[42,49] = -0.0982363113431474
        struct[0].Gy_ini[42,50] = -0.0293176867112776
        struct[0].Gy_ini[42,51] = 0.0982363113431526
        struct[0].Gy_ini[42,54] = 0.0293174777213170
        struct[0].Gy_ini[42,55] = -0.0982362237268541
        struct[0].Gy_ini[42,56] = -0.0293174777213155
        struct[0].Gy_ini[42,57] = 0.0982362237268593
        struct[0].Gy_ini[42,60] = 0.0293168507523162
        struct[0].Gy_ini[42,61] = -0.0982359608775054
        struct[0].Gy_ini[42,62] = -0.0293168507523146
        struct[0].Gy_ini[42,63] = 0.0982359608775105
        struct[0].Gy_ini[42,66] = 11.6029687203683
        struct[0].Gy_ini[42,67] = -3.08804231916211
        struct[0].Gy_ini[42,68] = 10.1951871226168
        struct[0].Gy_ini[42,69] = 1.79131033552716
        struct[0].Gy_ini[42,70] = 10.1951871226167
        struct[0].Gy_ini[42,71] = 1.79131033552690
        struct[0].Gy_ini[43,43] = -1
        struct[0].Gy_ini[43,48] = 0.0982363113431474
        struct[0].Gy_ini[43,49] = 0.0293176867112792
        struct[0].Gy_ini[43,50] = -0.0982363113431526
        struct[0].Gy_ini[43,51] = -0.0293176867112776
        struct[0].Gy_ini[43,54] = 0.0982362237268541
        struct[0].Gy_ini[43,55] = 0.0293174777213170
        struct[0].Gy_ini[43,56] = -0.0982362237268593
        struct[0].Gy_ini[43,57] = -0.0293174777213155
        struct[0].Gy_ini[43,60] = 0.0982359608775054
        struct[0].Gy_ini[43,61] = 0.0293168507523162
        struct[0].Gy_ini[43,62] = -0.0982359608775105
        struct[0].Gy_ini[43,63] = -0.0293168507523146
        struct[0].Gy_ini[43,66] = 3.08804231916211
        struct[0].Gy_ini[43,67] = 11.6029687203683
        struct[0].Gy_ini[43,68] = -1.79131033552716
        struct[0].Gy_ini[43,69] = 10.1951871226168
        struct[0].Gy_ini[43,70] = -1.79131033552690
        struct[0].Gy_ini[43,71] = 10.1951871226167
        struct[0].Gy_ini[44,44] = -1
        struct[0].Gy_ini[44,50] = 0.0293176867112812
        struct[0].Gy_ini[44,51] = -0.0982363113431524
        struct[0].Gy_ini[44,52] = -0.0293176867112826
        struct[0].Gy_ini[44,53] = 0.0982363113431489
        struct[0].Gy_ini[44,56] = 0.0293174777213190
        struct[0].Gy_ini[44,57] = -0.0982362237268590
        struct[0].Gy_ini[44,58] = -0.0293174777213204
        struct[0].Gy_ini[44,59] = 0.0982362237268555
        struct[0].Gy_ini[44,62] = 0.0293168507523179
        struct[0].Gy_ini[44,63] = -0.0982359608775103
        struct[0].Gy_ini[44,64] = -0.0293168507523194
        struct[0].Gy_ini[44,65] = 0.0982359608775067
        struct[0].Gy_ini[44,66] = 10.1951871226169
        struct[0].Gy_ini[44,67] = 1.79131033552732
        struct[0].Gy_ini[44,68] = 11.6029687203686
        struct[0].Gy_ini[44,69] = -3.08804231916193
        struct[0].Gy_ini[44,70] = 10.1951871226169
        struct[0].Gy_ini[44,71] = 1.79131033552715
        struct[0].Gy_ini[45,45] = -1
        struct[0].Gy_ini[45,50] = 0.0982363113431524
        struct[0].Gy_ini[45,51] = 0.0293176867112812
        struct[0].Gy_ini[45,52] = -0.0982363113431489
        struct[0].Gy_ini[45,53] = -0.0293176867112826
        struct[0].Gy_ini[45,56] = 0.0982362237268590
        struct[0].Gy_ini[45,57] = 0.0293174777213190
        struct[0].Gy_ini[45,58] = -0.0982362237268555
        struct[0].Gy_ini[45,59] = -0.0293174777213204
        struct[0].Gy_ini[45,62] = 0.0982359608775103
        struct[0].Gy_ini[45,63] = 0.0293168507523179
        struct[0].Gy_ini[45,64] = -0.0982359608775067
        struct[0].Gy_ini[45,65] = -0.0293168507523194
        struct[0].Gy_ini[45,66] = -1.79131033552732
        struct[0].Gy_ini[45,67] = 10.1951871226169
        struct[0].Gy_ini[45,68] = 3.08804231916193
        struct[0].Gy_ini[45,69] = 11.6029687203686
        struct[0].Gy_ini[45,70] = -1.79131033552715
        struct[0].Gy_ini[45,71] = 10.1951871226169
        struct[0].Gy_ini[46,46] = -1
        struct[0].Gy_ini[46,48] = -0.0293176867112787
        struct[0].Gy_ini[46,49] = 0.0982363113431550
        struct[0].Gy_ini[46,52] = 0.0293176867112779
        struct[0].Gy_ini[46,53] = -0.0982363113431576
        struct[0].Gy_ini[46,54] = -0.0293174777213165
        struct[0].Gy_ini[46,55] = 0.0982362237268616
        struct[0].Gy_ini[46,58] = 0.0293174777213157
        struct[0].Gy_ini[46,59] = -0.0982362237268642
        struct[0].Gy_ini[46,60] = -0.0293168507523155
        struct[0].Gy_ini[46,61] = 0.0982359608775129
        struct[0].Gy_ini[46,64] = 0.0293168507523147
        struct[0].Gy_ini[46,65] = -0.0982359608775155
        struct[0].Gy_ini[46,66] = 10.1951871226168
        struct[0].Gy_ini[46,67] = 1.79131033552721
        struct[0].Gy_ini[46,68] = 10.1951871226169
        struct[0].Gy_ini[46,69] = 1.79131033552733
        struct[0].Gy_ini[46,70] = 11.6029687203684
        struct[0].Gy_ini[46,71] = -3.08804231916218
        struct[0].Gy_ini[47,47] = -1
        struct[0].Gy_ini[47,48] = -0.0982363113431550
        struct[0].Gy_ini[47,49] = -0.0293176867112787
        struct[0].Gy_ini[47,52] = 0.0982363113431576
        struct[0].Gy_ini[47,53] = 0.0293176867112779
        struct[0].Gy_ini[47,54] = -0.0982362237268616
        struct[0].Gy_ini[47,55] = -0.0293174777213165
        struct[0].Gy_ini[47,58] = 0.0982362237268642
        struct[0].Gy_ini[47,59] = 0.0293174777213157
        struct[0].Gy_ini[47,60] = -0.0982359608775129
        struct[0].Gy_ini[47,61] = -0.0293168507523155
        struct[0].Gy_ini[47,64] = 0.0982359608775155
        struct[0].Gy_ini[47,65] = 0.0293168507523147
        struct[0].Gy_ini[47,66] = -1.79131033552721
        struct[0].Gy_ini[47,67] = 10.1951871226168
        struct[0].Gy_ini[47,68] = -1.79131033552733
        struct[0].Gy_ini[47,69] = 10.1951871226169
        struct[0].Gy_ini[47,70] = 3.08804231916218
        struct[0].Gy_ini[47,71] = 11.6029687203684
        struct[0].Gy_ini[48,0] = i_W1lv_a_r
        struct[0].Gy_ini[48,1] = i_W1lv_a_i
        struct[0].Gy_ini[48,48] = v_W1lv_a_r
        struct[0].Gy_ini[48,49] = v_W1lv_a_i
        struct[0].Gy_ini[49,2] = i_W1lv_b_r
        struct[0].Gy_ini[49,3] = i_W1lv_b_i
        struct[0].Gy_ini[49,50] = v_W1lv_b_r
        struct[0].Gy_ini[49,51] = v_W1lv_b_i
        struct[0].Gy_ini[50,4] = i_W1lv_c_r
        struct[0].Gy_ini[50,5] = i_W1lv_c_i
        struct[0].Gy_ini[50,52] = v_W1lv_c_r
        struct[0].Gy_ini[50,53] = v_W1lv_c_i
        struct[0].Gy_ini[51,0] = -i_W1lv_a_i
        struct[0].Gy_ini[51,1] = i_W1lv_a_r
        struct[0].Gy_ini[51,48] = v_W1lv_a_i
        struct[0].Gy_ini[51,49] = -v_W1lv_a_r
        struct[0].Gy_ini[52,2] = -i_W1lv_b_i
        struct[0].Gy_ini[52,3] = i_W1lv_b_r
        struct[0].Gy_ini[52,50] = v_W1lv_b_i
        struct[0].Gy_ini[52,51] = -v_W1lv_b_r
        struct[0].Gy_ini[53,4] = -i_W1lv_c_i
        struct[0].Gy_ini[53,5] = i_W1lv_c_r
        struct[0].Gy_ini[53,52] = v_W1lv_c_i
        struct[0].Gy_ini[53,53] = -v_W1lv_c_r
        struct[0].Gy_ini[54,6] = i_W2lv_a_r
        struct[0].Gy_ini[54,7] = i_W2lv_a_i
        struct[0].Gy_ini[54,54] = v_W2lv_a_r
        struct[0].Gy_ini[54,55] = v_W2lv_a_i
        struct[0].Gy_ini[55,8] = i_W2lv_b_r
        struct[0].Gy_ini[55,9] = i_W2lv_b_i
        struct[0].Gy_ini[55,56] = v_W2lv_b_r
        struct[0].Gy_ini[55,57] = v_W2lv_b_i
        struct[0].Gy_ini[56,10] = i_W2lv_c_r
        struct[0].Gy_ini[56,11] = i_W2lv_c_i
        struct[0].Gy_ini[56,58] = v_W2lv_c_r
        struct[0].Gy_ini[56,59] = v_W2lv_c_i
        struct[0].Gy_ini[57,6] = -i_W2lv_a_i
        struct[0].Gy_ini[57,7] = i_W2lv_a_r
        struct[0].Gy_ini[57,54] = v_W2lv_a_i
        struct[0].Gy_ini[57,55] = -v_W2lv_a_r
        struct[0].Gy_ini[58,8] = -i_W2lv_b_i
        struct[0].Gy_ini[58,9] = i_W2lv_b_r
        struct[0].Gy_ini[58,56] = v_W2lv_b_i
        struct[0].Gy_ini[58,57] = -v_W2lv_b_r
        struct[0].Gy_ini[59,10] = -i_W2lv_c_i
        struct[0].Gy_ini[59,11] = i_W2lv_c_r
        struct[0].Gy_ini[59,58] = v_W2lv_c_i
        struct[0].Gy_ini[59,59] = -v_W2lv_c_r
        struct[0].Gy_ini[60,12] = i_W3lv_a_r
        struct[0].Gy_ini[60,13] = i_W3lv_a_i
        struct[0].Gy_ini[60,60] = v_W3lv_a_r
        struct[0].Gy_ini[60,61] = v_W3lv_a_i
        struct[0].Gy_ini[61,14] = i_W3lv_b_r
        struct[0].Gy_ini[61,15] = i_W3lv_b_i
        struct[0].Gy_ini[61,62] = v_W3lv_b_r
        struct[0].Gy_ini[61,63] = v_W3lv_b_i
        struct[0].Gy_ini[62,16] = i_W3lv_c_r
        struct[0].Gy_ini[62,17] = i_W3lv_c_i
        struct[0].Gy_ini[62,64] = v_W3lv_c_r
        struct[0].Gy_ini[62,65] = v_W3lv_c_i
        struct[0].Gy_ini[63,12] = -i_W3lv_a_i
        struct[0].Gy_ini[63,13] = i_W3lv_a_r
        struct[0].Gy_ini[63,60] = v_W3lv_a_i
        struct[0].Gy_ini[63,61] = -v_W3lv_a_r
        struct[0].Gy_ini[64,14] = -i_W3lv_b_i
        struct[0].Gy_ini[64,15] = i_W3lv_b_r
        struct[0].Gy_ini[64,62] = v_W3lv_b_i
        struct[0].Gy_ini[64,63] = -v_W3lv_b_r
        struct[0].Gy_ini[65,16] = -i_W3lv_c_i
        struct[0].Gy_ini[65,17] = i_W3lv_c_r
        struct[0].Gy_ini[65,64] = v_W3lv_c_i
        struct[0].Gy_ini[65,65] = -v_W3lv_c_r
        struct[0].Gy_ini[66,18] = i_POImv_a_r
        struct[0].Gy_ini[66,19] = i_POImv_a_i
        struct[0].Gy_ini[66,66] = v_POImv_a_r
        struct[0].Gy_ini[66,67] = v_POImv_a_i
        struct[0].Gy_ini[67,20] = i_POImv_b_r
        struct[0].Gy_ini[67,21] = i_POImv_b_i
        struct[0].Gy_ini[67,68] = v_POImv_b_r
        struct[0].Gy_ini[67,69] = v_POImv_b_i
        struct[0].Gy_ini[68,22] = i_POImv_c_r
        struct[0].Gy_ini[68,23] = i_POImv_c_i
        struct[0].Gy_ini[68,70] = v_POImv_c_r
        struct[0].Gy_ini[68,71] = v_POImv_c_i
        struct[0].Gy_ini[69,18] = -i_POImv_a_i
        struct[0].Gy_ini[69,19] = i_POImv_a_r
        struct[0].Gy_ini[69,66] = v_POImv_a_i
        struct[0].Gy_ini[69,67] = -v_POImv_a_r
        struct[0].Gy_ini[70,20] = -i_POImv_b_i
        struct[0].Gy_ini[70,21] = i_POImv_b_r
        struct[0].Gy_ini[70,68] = v_POImv_b_i
        struct[0].Gy_ini[70,69] = -v_POImv_b_r
        struct[0].Gy_ini[71,22] = -i_POImv_c_i
        struct[0].Gy_ini[71,23] = i_POImv_c_r
        struct[0].Gy_ini[71,70] = v_POImv_c_i
        struct[0].Gy_ini[71,71] = -v_POImv_c_r





@numba.njit(cache=True)
def Piecewise(arg):
    out = arg[0][1]
    N = len(arg)
    for it in range(N-1,-1,-1):
        if arg[it][1]: out = arg[it][0]
    return out

@numba.njit(cache=True)
def ITE(arg):
    out = arg[0][1]
    N = len(arg)
    for it in range(N-1,-1,-1):
        if arg[it][1]: out = arg[it][0]
    return out


@numba.njit(cache=True)
def Abs(x):
    return np.abs(x)



@numba.njit(cache=True) 
def daesolver(struct): 
    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    i = 0 
    
    Dt = struct[i].Dt 

    N_x = struct[i].N_x
    N_y = struct[i].N_y
    N_z = struct[i].N_z

    decimation = struct[i].decimation 
    eye = np.eye(N_x)
    t = struct[i].t 
    t_end = struct[i].t_end 
    if struct[i].it == 0:
        run(t,struct, 1) 
        struct[i].it_store = 0  
        struct[i]['T'][0] = t 
        struct[i].X[0,:] = struct[i].x[:,0]  
        struct[i].Y[0,:] = struct[i].y_run[:,0]  
        struct[i].Z[0,:] = struct[i].h[:,0]  

    solver = struct[i].solvern 
    while t<t_end: 
        struct[i].it += 1
        struct[i].t += Dt
        
        t = struct[i].t


            
        if solver == 5: # Teapezoidal DAE as in Milano's book

            run(t,struct, 2) 
            run(t,struct, 3) 

            x = np.copy(struct[i].x[:]) 
            y = np.copy(struct[i].y_run[:]) 
            f = np.copy(struct[i].f[:]) 
            g = np.copy(struct[i].g[:]) 
            
            for iter in range(struct[i].imax):
                run(t,struct, 2) 
                run(t,struct, 3) 
                run(t,struct,10) 
                run(t,struct,11) 
                
                x_i = struct[i].x[:] 
                y_i = struct[i].y_run[:]  
                f_i = struct[i].f[:] 
                g_i = struct[i].g[:]                 
                F_x_i = struct[i].Fx[:,:]
                F_y_i = struct[i].Fy[:,:] 
                G_x_i = struct[i].Gx[:,:] 
                G_y_i = struct[i].Gy[:,:]                

                A_c_i = np.vstack((np.hstack((eye-0.5*Dt*F_x_i, -0.5*Dt*F_y_i)),
                                   np.hstack((G_x_i,         G_y_i))))
                     
                f_n_i = x_i - x - 0.5*Dt*(f_i+f) 
                # print(t,iter,g_i)
                Dxy_i = np.linalg.solve(-A_c_i,np.vstack((f_n_i,g_i))) 
                
                x_i = x_i + Dxy_i[0:N_x]
                y_i = y_i + Dxy_i[N_x:(N_x+N_y)]

                struct[i].x[:] = x_i
                struct[i].y_run[:] = y_i

        # [f_i,g_i,F_x_i,F_y_i,G_x_i,G_y_i] =  smib_transient(x_i,y_i,u);
        
        # A_c_i = [[eye(N_x)-0.5*Dt*F_x_i, -0.5*Dt*F_y_i],
        #          [                G_x_i,         G_y_i]];
             
        # f_n_i = x_i - x - 0.5*Dt*(f_i+f);
        
        # Dxy_i = -A_c_i\[f_n_i.',g_i.'].';
        
        # x_i = x_i + Dxy_i(1:N_x);
        # y_i = y_i + Dxy_i(N_x+1:N_x+N_y);
                
                xy = np.vstack((x_i,y_i))
                max_relative = 0.0
                for it_var in range(N_x+N_y):
                    abs_value = np.abs(xy[it_var,0])
                    if abs_value < 0.001:
                        abs_value = 0.001
                                             
                    relative_error = np.abs(Dxy_i[it_var,0])/abs_value
                    
                    if relative_error > max_relative: max_relative = relative_error
                    
                if max_relative<struct[i].itol:
                    
                    break
                
                # if iter>struct[i].imax-2:
                    
                #     print('Convergence problem')

            struct[i].x[:] = x_i
            struct[i].y_run[:] = y_i
                
        # channels 
        it_store = struct[i].it_store
        if struct[i].it >= it_store*decimation: 
            struct[i]['T'][it_store+1] = t 
            struct[i].X[it_store+1,:] = struct[i].x[:,0] 
            struct[i].Y[it_store+1,:] = struct[i].y_run[:,0]
            struct[i].Z[it_store+1,:] = struct[i].h[:,0]
            struct[i].iters[it_store+1,0] = iter
            struct[i].it_store += 1 
            
    struct[i].t = t

    return t


