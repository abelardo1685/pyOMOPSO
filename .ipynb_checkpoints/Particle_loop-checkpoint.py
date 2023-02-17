import ray # 
import numpy as np
from mutate import mutate
from Dominate import Dominate
from Dominate_best import dominate_best
from Dominate_new import dominate_new

from copy import deepcopy

def particle_loop(leader, nPop, problem, pop, OMOPSO, it):
   # mutation.setflags(write=True)
    # ray.shutdown()
    # ray.init()

    # Define the function. Each remote function will be executed 
    # in a separate process.
    @ray.remote
    def particle_iteration(leader, nPop, problem, pop, OMOPSO, it):
      
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
 
        for i in range(0, nPop):

            W = (w2 - w1) * np.random.uniform(0,1,nVar) + w1

            # 1.2.- Calculo de coeficientes de aprendizaje
            C1 = (c2 - c1) * np.random.uniform(0,1,nVar) + c1
            C2 = (c2 - c1) * np.random.uniform(0,1,nVar) + c1

            R1 = (r2 - r1) * np.random.uniform(0,1,nVar) + r1
            R2 = (r2 - r1) * np.random.uniform(0,1,nVar) + r1

            pop[i]['velocity'] = W * pop[i]['velocity'].copy() + \
                (R1 * C1 * (pop[i]['best_position'].copy()-pop[i]['position'].copy())) + \
                (R2 * C2 * (leader['position'].copy()-pop[i]['position'].copy()))       

            pop[i]['position'] =  pop[i]['position'] +  pop[i]['velocity']
            pop[i]['position'] =  np.fmax(pop[i]['position'], VarMin)                            
            pop[i]['position'] =  np.fmin(pop[i]['position'], VarMax)                            

            pop[i]['cost'] = CostFunction(pop[i]['position'])


        return pop
    
    output_ids = []
    for i in range(1):
        output_ids.append(particle_iteration.remote(leader, nPop, problem, pop, OMOPSO, it))

    # Get results when ready.
    output_list = ray.get(output_ids)
    output = output_list[0]
    
    return output
