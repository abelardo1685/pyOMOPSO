"""

Copyright (c) 2022, Abelardo Rodríguez Pretelín
All rights reserved. Please read the "license.txt" for usage terms.
__________________________________________________________________________

Project Code: parOMOPSO
Project Title: Implementation of a parallel version of the Optimal multi-objective particle swarm optimizer in Python. 
According to [2], OMOPSO is one of the top-performing PSO algorithms.
References:

[1] Sierra, M. R. and C. A. Coello Coello (2005). Improving PSO-based Multi-Objective Optimization using Crowding, Mutation and ε-Dominance. Evolutionary Multi-Criterion Optimization, pp. 505-519.
[2] Durillo, J. J., J. Garcia-Nieto, A. J. Nebro, C. A. Coello Coello, F. Luna, and E. Alba (2009). Multi-Objective Particle Swarm Optimizers: An Experimental Comparison. Evolutionary Multi-Criterion Optimization, pp. 495-509.

Developer: A. Rodríguez-Pretelín (Head of IA Team)

Contact Info: abelardo.rodriguez.pretelin@gmail.com """

## General library
import numpy as np

## Internal library
from Particle_creation import particle_creation
from Particle_loop import particle_loop

from Test_Functions import *
from Mat_cost import *
from Mat_pos import *
from Mat_grid_index import *
from Mat_dom import *

from Sel_Leader import SelectLeader
from mutate import mutate
from Dominate import Dominate
from Crow_leaders import crowding_leaders
from Dominate_rep import DominateRepository
from Dominate_best import dominate_best
from Dominate_new import dominate_new
from Rep_dominance import Repository_dominance
from Dominate_pop import Dominate_pop
from Domination import determine_domination
from copy import deepcopy
from cre_grid import CreateGrid
from FGI import FindGridIndex
    
def parOMOPSO(problem, OMOPSO):

    # Retrieving values from problem and OMOPSO dictionaries

    VarMin = problem['VarMin']
    VarMax = problem['VarMax']
    nVar   = problem['nVar']
    CostFunction = problem['CostFunction']

    MaxIter = OMOPSO['MaxIter']
    nPop = OMOPSO['nPop']
    w1 = OMOPSO['w1']
    w2 = OMOPSO['w2']
    e = OMOPSO['e']
    c1 = OMOPSO['c1']
    c2 = OMOPSO['c2']
    nGrid = OMOPSO['nGrid']
    alpha = OMOPSO['alpha']
    wdamp = OMOPSO['wdamp']
    r1 = OMOPSO['r1']
    r2 = OMOPSO['r2']


    # Initialization: Empty Particle Template
    empty_particle = {
        'position': None,
        'velocity': None,
        'cost': None,
        'best_position': None,
        'best_cost': None,
        'is_dominated': None,
        'grid_index': None,
        'grid_subindex': None,
        'crowding_distance': None,
        'New_pos': None,
        'New_cost': None,
        'Mutation': None,
        };

    e_file  = []

    ### Creation of the particle population and domination analysis and preparation of pop before main loop
    pop = particle_creation(problem, empty_particle, OMOPSO)

    # Define which particles dominates over the others
    pop = determine_domination(pop)

    rep = []

    for i in range(0, len(pop)):
        if pop[i]['is_dominated'] == 0:
            rep.append(deepcopy(pop[i]))

    # Grid=CreateGrid(pop,nGrid,alpha)
    Grid_LB, Grid_UB = CreateGrid(rep, OMOPSO)

    for i in range(0,len(rep)):
        rep[i] = FindGridIndex(rep[i], Grid_LB, Grid_UB)


    # MAIN parOMOPSO loop:
    # MaxIter: Number of movements of the particle swarm
    # pop: Group of particles.    

    for it in range (0, MaxIter): 

        mutation = np.zeros((nPop,1))
        leader, rep = SelectLeader(rep,nPop)

        pop = particle_loop(leader, nPop, problem, pop, OMOPSO, it)

        mu = 1/nVar
        a  = 1 
        pm = (1-(it)/(MaxIter-1))**(a/mu)

        for i in range(0, nPop):

            NewSol, mutation = mutate(pop[i]['position'].copy(), pm, VarMin, VarMax, i, mutation)
            NewSol['cost'] =  deepcopy(CostFunction(NewSol['position'].copy()))

            if Dominate(NewSol, pop[i]):
                pop[i]['position'] = NewSol['position'].copy()
                pop[i]['cost'] =  NewSol['cost'].copy()

            if dominate_best(pop[i], pop[i]['best_cost'].copy()):
                pop[i]['best_position'] = pop[i]['position'].copy()
                pop[i]['best_cost'] = pop[i]['cost'].copy()

         # Outer Loop. Contrast between iterations
        # 5.0 Add Non-Dominated Particles to REPOSITORY
        for i in range(0,len(pop)):
            if pop[i]['is_dominated'] == 0:
                rep.append(pop[i].copy())

        #5.1- Update leaders: if the repository size > swarm size keep only the best measuring crowding values Pareto comparison
        ## 5.1.- Determine Domination of New Resository Members
        rep = determine_domination(rep)

        ## 5.2.- Keep only Non-Dominated Memebrs in the Repository
        rep = [sub for sub in rep if sub['is_dominated'] == 0]

        ## 5.3.- Crowding distance  
        if len(rep) > nPop:
            rep = crowding_leaders (rep, nPop)    

        ## 5.4.- Update Grid
        Grid_LB, Grid_UB = CreateGrid(rep, OMOPSO)

        ## 5.5.- Update Grid Indices
        for i in range(0,len(rep)):
            rep[i] = FindGridIndex(rep[i], Grid_LB, Grid_UB)

        ## 6.- Update e final values
        # Uso of e dominance: Contain the file with the non dominant solutions reported by the algoritm
        e_file, rep = Repository_dominance(rep,e_file,e)
        print('Iteration {}: '.format(it))
    print('Iteration {}: '.format(MaxIter))
    return e_file, rep;