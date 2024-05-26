import numpy as np
import random

def infect_step(G,p1,individuals,N):
    '''The function serves as the infection model for each day.
    input params:
    G (ndarray N*N): the adjacency matrix.
    p1: the probability each individual infects neighbours.
    '''

    individuals_updated = np.copy(individuals)
    for i in range(N):
        if individuals[i] == 1:  # Check if individual i is infected
            for j in range(N):
                if G[i, j] == 1 and individuals[j] == 0:  # If there's a connection and j is not yet infected
                    if random.random() < p1:  # With probability p1, infect neighbor
                        individuals_updated[j] = 1
    return individuals_updated




def infect(G,p0,p1,time_steps):
    '''The function serves as the infection model for each day.
    input params (consistent with the project description):
    G (ndarray N*N): the adjacency matrix.
    p0: the infection probability for initial status.
    p1: the probability each individual infects neighbours.
    time_steps: log N
    '''
    N = G.shape[0]
    individuals = np.zeros(N)
    ###################################################
    # Initially infect some individuals randomly based on p0
    for i in range(N):
        if random.random() < p0:
            individuals[i] = 1
    
    for _ in range(time_steps):
        individuals = infect_step(G, p1, individuals, N)
    
    return individuals