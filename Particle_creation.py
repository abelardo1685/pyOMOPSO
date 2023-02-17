import ray
import numpy as np

def particle_creation(problem, empty_particle, OMOPSO):
    
    nPop = OMOPSO['nPop']
    ray.shutdown()
    ray.init()

    # Define the function. Each remote function will be executed 
    # in a separate process.
    @ray.remote
    def particle_create(problem, empty_particle):

        VarMin = problem['VarMin']
        VarMax = problem['VarMax']
        nVar   = problem['nVar']
        CostFunction = problem['CostFunction']
                
        pop = empty_particle.copy()

        pop['position'] = np.random.uniform(VarMin, VarMax, nVar)
        pop['velocity'] = np.zeros(nVar)
        pop['cost'] = CostFunction(pop['position']);
        pop['best_position'] = pop['position'].copy();
        pop['best_cost'] = pop['cost']

        return pop

    output_ids = []
    for i in range(nPop):
        output_ids.append(particle_create.remote(problem, empty_particle))

    # Get results when ready.
    output_list = ray.get(output_ids)

    return output_list