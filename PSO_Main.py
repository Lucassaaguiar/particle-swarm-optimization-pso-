##############################################################################
#                         Particle Swarm Optimizers                          #
#----------------------------------------------------------------------------#
# Author: Eng. Msc. Lucas Alves de Aguiar                                    #
# Date: 31/08/2023                                                           #
# Last Review: 04/08//2024                                                   #
##############################################################################

from PSO_iteration import *
import numpy as np

##################################################################
### ------------------- Problem Definiton -------------------- ###
##################################################################

### -------- Objective Function (Weight Minimization) -------- ###

def objective(x):
    x1, x2 = x      
    # f      = 2*x1**2 - 1.04*x1**4 + (x1**6)/6 + x1*x2 + x2**2                        # Three-hump camel function
    # f      = -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2)) # Easom function
    # f      = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) + \
    #             (-np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))) \
    #             + np.e + 20                                                           # Ackley function
    f        = (np.sin(x2) * np.exp((1 - np.cos(x1))**2) +
                np.cos(x1) * np.exp((1 - np.sin(x2))**2) +
                (x1 - 2)**2)                                                            # Mishra's Bird function - constrained
    return f

### ---------------------- Constraints ----------------------- ###

def g1(x):
    x1, x2 = x
    g = ((x1 + 5)**2 + (x2 + 5)**2) - 25
    return g
def g2(x):
    x1, x2 = x
    g = 0 
    return g

### ------------------ PSO Initials Params ------------------- ###
nVar        =   2                      # Number of Unknown (Decision) Variables
VarMin      =   np.array([-10, -6.5])       # Lower Bound of Decision Variable 
VarMax      =   np.array([0, 0])       # Upper Bound of Decision Variable
MaxIt       =   5000                    # Maximum Number of Interations
nPop        =   500                    
 # Population Size (Swarm Size)  
stop        =   200                    # Iteration Stop Criterion      
constraints = np.array([g1])

### -------------------------- Main -------------------------- ###
x,y,z,w,k = PSO(nVar, VarMin, VarMax, MaxIt, nPop, constraints, stop, objective)

### -------- Results -------- ###
print("X: {0:5.2f} []".format(x[0]))
print("Y: {0:5.2f} []".format(x[1]))
print("f(x): {0:5.2f} []".format(k))  

# Ploting a iteration graph
plt.figure(1,figsize=(9,5))                         
plt.grid(True,zorder=0)                             
plt.xlabel('Iteration number', fontsize = 14)        
plt.ylabel('Cost Fuction [f(x)]', fontsize = 14)       
plt.plot(y, lw = '1.7', color = '#00008B')

plt.show

# Plot Saving
#plt.savefig('Convergence Graph.svg', backend=None)