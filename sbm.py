import numpy as np
import random

def SBM(N, M, q0, q1):
    '''This function is designed to generate the Stochastic Block Model.
    input params:
    N (int): Total number of nodes.
    M (int): Number of communities.
    q0 (float): Probability of intra-community connections.
    q1 (float): Probability of inter-community connections.

    output:
    G (N*N): adjacency matrix of the generated graph.
    '''
    
    # Step 1: Assign nodes to communities
    community_membership = np.random.choice(M, N)
    # np.random.choice(M, N) will generate an array of length N where each element 
    # is a random integer between 0 and M−1. This effectively assigns each node to 
    # one of M communities.
    
    
    # Step 2: Initialize the adjacency matrix with all zeros
    G = np.zeros((N, N), dtype=int)
    
    # Step 3: Generate edges based on probabilities
    for i in range(N):
        for j in range(i + 1, N): #this is to make sure each pair of nodes (i,j) is considered only once
            if community_membership[i] == community_membership[j]:
                # Nodes i and j are in the same community with prob q0 they are connected
                if random.random() < q0:
                    G[i, j] = 1
                    G[j, i] = 1  # For undirected graph
            else:
                # Nodes i and j are in different communities with prob q1 they are connected
                if random.random() < q1: 
                    G[i, j] = 1
                    G[j, i] = 1  # For undirected graph

    return G

    
    # # Initialize the adjacency matrix
    # G = np.zeros((N, N))

    # # Calculate the size of each community
    # sizes = [N // M + (1 if x < N % M else 0) for x in range(M)]
    
    # # Start index of each community
    # start_indices = np.cumsum([0] + sizes[:-1])
    
    # # Populate the adjacency matrix
    # for i in range(M):
    #     start_i = start_indices[i]
    #     end_i = start_i + sizes[i]
        
    #     for j in range(M):
    #         start_j = start_indices[j]
    #         end_j = start_j + sizes[j]
            
    #         if i == j:
    #             # Nodes within the same community
    #             G[start_i:end_i, start_j:end_j] = np.random.rand(sizes[i], sizes[j]) < q0
    #         else:
    #             # Nodes in different communities
    #             G[start_i:end_i, start_j:end_j] = np.random.rand(sizes[i], sizes[j]) < q1
    
    # # Ensure the matrix is symmetric and has no self-loops
    # np.fill_diagonal(G, 0)
    # G = np.triu(G) + np.triu(G, 1).T

    # return G
