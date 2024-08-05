##############################################################################
#                         Particle Swarm Optimizers                          #
#----------------------------------------------------------------------------#
# Author: Eng. Msc. Lucas Alves de Aguiar                                    #
# Date: 31/08/2023                                                           #
# Last Review: 04/08//2024                                                   #
##############################################################################

import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib
from matplotlib import rc
rc('font',**{'family':'Times New Roman','size' :12})
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rc('text', usetex=True)
matplotlib.rc('axes', labelsize=14)
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('lines', lw=1.0,color='k')
matplotlib.rc('axes',lw=0.75)
matplotlib.rc('legend', fontsize=12)

def PSO(nVar, VarMin, VarMax, MaxIt, nPop, constraints, stop, f):
    
    def cost(constraints, f, xo1,k):
        xo         = xo1        
        # Quadratic Penalty due to constraint violation   
        for i in range(len(constraints)):
            pen    = 0
            pen    = np.sum([np.maximum(constraints[i](xo),0)**2]) * 1000
        
        if len(constraints) == 0:
            pen    = 0
        
        CostFunction = f(xo) + pen

        return CostFunction

    # Parameters of PSO (Clerc e Kennedy (2002))
    phi1        = 2.01
    phi2        = 2.01
    kappa       = 1
    phi         = phi1 + phi2
    chi         = 2*kappa/abs(2-phi-np.sqrt(phi**2-4*phi))
    w           = chi       # Intertia Coefficient
    wdamp       = 0.98      # Damping ratio of inertia Coefficient
    c1          = chi*phi1  # Personal Acceleration Coefficient
    c2          = chi*phi2  # Social Acceleration Coefficient
    
    max_vel     = 0.2*(np.amax(VarMax-VarMin))
    min_vel     = -max_vel

    # Initialization

    # The Particle Template
    class Particle:
        def __init__(self):
            self.position = None
            self.velocity = None
            self.cost     = None
            self.pen      = None

    class Part_Best:
        def __init__(self):
            self.position = None
            self.cost     = None
            self.pen      = None

    # Global Best
    class GlobalBest:
        def __init__(self):
            self.cost     = 10e100
            self.position = None
            self.pen      = None
            self.count    = 0

    empty_particle          = Particle()
    empty_part_best         = Part_Best()
    global_best             = GlobalBest()

    # Create Population Array
    particle = [empty_particle] * nPop         
    part_best= [empty_part_best] * nPop

    # Initialize Population Members
    for i in range(nPop):
        # Generate Random Solution
        particle[i].position = VarMin + (VarMax-VarMin) * np.random.uniform(0,1,nVar)
        # Initialize Velocity
        particle[i].velocity = np.zeros(nVar)
        # Evaluation
        particle[i].cost = cost(constraints, f, particle[i].position,0)
        # Update the Personal Best
        part_best[i].position = particle[i].position
        part_best[i].cost     = particle[i].cost
        part_best[i].pen      = particle[i].pen
        # Update Global Best 
        if part_best[i].cost < global_best.cost:
            global_best.cost     = part_best[i].cost
            global_best.Position = part_best[i].position
            global_best.pen      = part_best[i].pen        
    
    print(f'Best Initial Iteration {i+1}: Best Cost = {global_best.cost}')
            
    best_costs = np.ones((MaxIt, 1))*np.NaN
    printcounter = -1
    # Main Loop of PSO
    for it in range(MaxIt):
        global_best.count    += 1
        printcounter += 1
        if global_best.count > stop:
            break
        for i in range(nPop):
            # Update Velocity
            particle[i].velocity = (w*particle[i].velocity + c1*np.random.rand(nVar)*(part_best[i].position - 
                                   particle[i].position) + c2*np.random.rand(nVar)*(global_best.Position - 
                                   particle[i].position))
                                                                                   
            # Velocity Limits
            particle[i].velocity = np.maximum(particle[i].velocity, min_vel)
            particle[i].velocity = np.minimum(particle[i].velocity, max_vel)                                                                           
                                                                                    
            # Update position
            particle[i].position    = particle[i].position + particle[i].velocity

            # Apply Lower and Upper Bound Limits 
            particle[i].position = np.maximum(particle[i].position, VarMin)
            particle[i].position = np.minimum(particle[i].position, VarMax)
            # Evaluation
            particle[i].cost  = cost(constraints, f, particle[i].position,it)
            # Update the Personal Best
            if particle[i].cost < part_best[i].cost:
                part_best[i].Position = particle[i].position
                part_best[i].cost     = particle[i].cost
                part_best[i].pen      = particle[i].pen
                # Update Global Best 
                if part_best[i].cost < global_best.cost:
                    global_best.cost     = part_best[i].cost
                    global_best.Position = part_best[i].Position
                    global_best.pen      = part_best[i].pen
                    global_best.count    = 0
                    
        best_costs[it] = global_best.cost
        # Display Iteration Information 
        if (printcounter == 20):
            print(f'iteration {it}: Best Cost = {best_costs[it]}')
            # print(global_best.Position)
            printcounter = 0
        # Damping Inertia Coefficient
        w = w * wdamp  

    return global_best.Position, best_costs, part_best, global_best.pen, global_best.cost

