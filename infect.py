import numpy as np
import random

def infect_step(G,p1,individuals,N):
    '''The function serves as the infection model for each day.
    input params:
    G (ndarray N*N): the adjacency matrix.
    p1: the probability each individual infects neighbours.
    '''
    
    # Process: 
    # Here we use the given adjacency matrix to incorporate the two setep 
    # infection. We will step through all elements of the matrix and see 
    # which individuals are connected. If connected to anyone (i.e. G_ij = 1)
    # then with probability p1 we can infect them, otherwise move on. 
    
    individuals_updated = np.copy(individuals) #make a copy of the individuals array
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
    N = G.shape[0] #get the number of rows (i.e. N from G)
    individuals = np.zeros(N) #initially everyone is not infected
    ###################################################
    # Initially infect some individuals randomly based on p0
    for i in range(N):
        if random.random() < p0:
            individuals[i] = 1
    
    # Now call the 2-step infection to also infect neighbors with prob  p1
    for _ in range(time_steps):
        individuals = infect_step(G, p1, individuals, N)
    
    return individuals