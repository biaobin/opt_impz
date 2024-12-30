
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import shutil
import subprocess

# Define the optimization problem
class MyProblem(Problem):
    def __init__(self):
        # 8 decision variables, 3 objectives, no constraints, and variable bounds [-10, 10]
        super().__init__(n_var=8,  # Number of variables
                         n_obj=3,  # Number of objectives
                         n_constr=0,  # Number of constraints
                         xl=np.array([-40] * 8),  # Lower bounds for variables
                         xu=np.array([40] * 8))  # Upper bounds for variables
                 
        self.np = 1e4   #total macro-particle number
        self.cnt = 0

    def _evaluate(self, x, out, *args, **kwargs):
        # update the simulation results
        self.update_simulation(x)
        
        # get simu results
        g1,f1,f2 = self.get_simu_results()
        
        print("g1=",g1)
        print("f1=",f1)
        print("f2=",f2)
        # Combine objectives into a single output array
        
        #out["G"] = g1  #match 
        out["F"] = np.column_stack([g1, f1, f2])  #eny & particle loss rate

    def update_simulation(self,x):
        # K1 = x
        # K1 = [24,-35,6]
        # change quad values from lte.impz
        fname  = 'lte.impz'   
        alt_q = ["PST_QT5","PST_QT6","HIGH2_Q1","HIGH2_Q2","HIGH2_Q5","High3_Q1","High3_Q2","High3_Q3"]    
        
        self.pop_size = x.shape[0]
        
        #(1). we need to mkdir pop_size folder for the first run
        #=======================================================
        print("self.cnt=",self.cnt)
        if self.cnt == 0:
            print("copying files from ini_simu")
            for j in range(self.pop_size):
                foldername = "simu_"+str(j+1)
                os.makedirs(foldername, exist_ok=True)
                
                # copy input files from ini_simu
                # ------------------------------
                base_from = "./ini_simu/"
                base_dest  = foldername+"/"
                files = ["lte.impz","particle.in","one","cleanUp"]
                for file in files: 
                    src_from = base_from +file
                    src_dest = base_dest +file
                
                    #copy everything, for the first iteration
                    shutil.copy(src_from, src_dest)
                        
                    # only copy non-exist filles  
                    # if not os.path.exists(src_dest):
                        # shutil.copy(src_from, src_dest) 
            self.cnt +=1
            
        #(2). update lattice for every iteration
        #====================================================
        for j in range(self.pop_size):    
            base_dest  = "simu_"+str(j+1)+"/"
            # update lte.impz with input-x
            # ------------------------------
            fname = base_dest+"lte.impz"
            
            with open(fname,'r') as f:
                lines = [line.lower() for line in f.readlines()]
            
            for ii,elem in enumerate(alt_q):
                for jj,line in enumerate(lines):
                    if elem.lower() in line.split(':'):
                        tmp = line.split(",")
                        tmp[-1]="K1="+str(x[j,ii])+'\n'
                        lines[jj] = ','.join(tmp)
                        # print(lines[jj])            
            # back to lte.impz
            with open(fname,'w') as f:
                f.writelines(line for line in lines)           
            print(fname+" is updated.")     

        #(3). Now is ready to run all the simulations simultaneously
        #=======================================================
        processes = []
        for j in range(self.pop_size):
            base_dest  = "simu_"+str(j+1)+"/"
            os.system(f"cd {base_dest} && bash cleanUp")  #cleanUp outputs

            process = subprocess.Popen(["bash", 'one'], cwd=base_dest)
            processes.append(process)

        # Wait for all processes to complete
        j= 0
        for process in processes:
            base_dest  = "simu_"+str(j+1)+"/"
            process.wait() #wait all simulations to be finished

            if process.returncode == 0:
                print(f"Process in folder {"simu_"+str(j+1)} completed.")

            else:
                print("Error for the simulation with the para settings in simu_"+str(j+1))
                #touch a file to indicate the error
                with open(base_dest+"RunError.flag","w"):
                    print("RunError.flag is touched in simu_"+str(j+1))
                    pass
            j+=1        
       
    def get_simu_results(self, debug="OFF", path='.'):
        fval1 = []
        fval2 = []
        fval3 = []
        
        if debug != "OFF":
            pop_size = 1
        else:
            pop_size = self.pop_size
        
        for j in range(pop_size):
            if debug !="OFF":
                base_dest = path+"/"
            else:
                base_dest = "simu_"+str(j+1)+"/"
            
            if not os.path.exists(base_dest+"RunError.flag"):
                # match beam to given twiss para
                twix = np.loadtxt(base_dest+'fort.24')
                twiy = np.loadtxt(base_dest+'fort.25')
                partNum = np.loadtxt(base_dest+'fort.28')
                
                twiss = {}
                twiss["betax"]  = twix[-1,8-1]
                twiss["alphax"] = twix[-1,6-1]
                twiss["betay"]  = twiy[-1,8-1]
                twiss["alphay"] = twiy[-1,6-1]
                
                twiss["enx"] = twix[-1,6]
                twiss["eny"] = twiy[-1,6]
                
                # target beta function
                betax = 8
                alphax = 4.55
                betay = 0.4
                alphay = 2.5
            
                f1  = self.sene(twiss["betax"],betax,1)
                f1 += self.sene(twiss["betay"],betay,1)
                f1 += self.sene(twiss["alphax"],alphax,0.1)
                f1 += self.sene(twiss["alphay"],alphay,0.1)
                   
                #f11 = (twiss["betax"]-betax)**2 +(twiss["alphax"]-alphax)**2
                #f12 = (twiss["betay"]-betay)**2 +(twiss["alphay"]-alphay)**2
    
                # get enx and eny
                f2 = twiss["eny"]*1e6 #um rad  #(twiss["enx"]+twiss["eny"])/2
                
                # particle loss rate
                f3 = (self.np-partNum[-1,3])/self.np
                
                #fval1.append(f1-1)  #if constraints, f1-1<=0
                fval1.append(f1)     # treat f1 as opt func
                fval2.append(f2)
                fval3.append(f3)
            else:
                print("IMPZ error, inf is given.")
                fval1.append(float('inf'))
                fval2.append(float('inf'))
                fval3.append(float('inf'))
                     
        return fval1,fval2,fval3
    
    def sene(self,V1,V2,T):
        if V1 > V2:    
            ff = ((V1-(V2+T))/T)**2
        else:
            ff = ((V2-(V1+T))/T)**2  
        
        return ff
        
# Configure the NSGA-II algorithm
#=========================================
base_folder = r'./'
os.chdir(base_folder)

npop  = 20
niter = 10

algorithm = NSGA2(
    pop_size=npop,  # Population size
    sampling=FloatRandomSampling(),  # Random initialization
    crossover=SBX(prob=0.9, eta=15),  # Simulated Binary Crossover (SBX)
    mutation=PM(eta=20),  # Polynomial Mutation
    eliminate_duplicates=True  # Avoid duplicate individuals
)

# Create the figure and axis for the 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# Perform the optimization
res = minimize(
    MyProblem(),  # Optimization problem
    algorithm,  # Algorithm
    termination=('n_gen', niter),  # Terminate after 200 generations
    seed=1,  # Set random seed for reproducibility
    verbose=True  # Print optimization progress
)

# Print results
print("Final Pareto Front (Objective Values):")
print(res.F)  # Objective values of Pareto-optimal solutions
print("Final Decision Variables:")
print(res.X)  # Decision variables corresponding to Pareto-optimal solutions

#save the results
with open("res.F","w") as f:
    np.savetxt(f, res.F, fmt="%15.6e")

with open("res.X","w") as f:
    np.savetxt(f, res.X, fmt="%15.6e")


## Plot the Pareto front in 3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(res.F[:, 0], res.F[:, 1], res.F[:, 2], c='red')
#ax.set_xlabel("match error")
#ax.set_ylabel("eny (um rad)")
#ax.set_zlabel("loss rate")
#plt.title("Pareto Front")
#plt.show()
#
##%%
## plt.figure()
## plt.plot(res.F[:,0],res.F[:,2],'.')
## plt.show()
#
#optrelt = (res.F[:,0] < 4) * (res.F[:,2] < 0.5e-6)
#
#res.F[optrelt,:]
#
#optX = res.X[optrelt,:]
#
#tmp = MyProblem()
#tmp.update_simulation(optX)

#%%
# plt.figure()
# plt.plot(res.F[:,0],res.F[:,2],'.')
# plt.show()

#%% debug
# tmp = MyProblem()

# path = base_folder +'/ini_simu'
# tmp.get_simu_results(debug="ON",path=path)
# # Out[74]: ([0.4141675194760004], [30.7898], [0.0])
