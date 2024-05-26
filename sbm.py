import numpy as np

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
    # Initialize the adjacency matrix
    G = np.zeros((N, N))

    # Calculate the size of each community
    sizes = [N // M + (1 if x < N % M else 0) for x in range(M)]
    
    # Start index of each community
    start_indices = np.cumsum([0] + sizes[:-1])
    
    # Populate the adjacency matrix
    for i in range(M):
        start_i = start_indices[i]
        end_i = start_i + sizes[i]
        
        for j in range(M):
            start_j = start_indices[j]
            end_j = start_j + sizes[j]
            
            if i == j:
                # Nodes within the same community
                G[start_i:end_i, start_j:end_j] = np.random.rand(sizes[i], sizes[j]) < q0
            else:
                # Nodes in different communities
                G[start_i:end_i, start_j:end_j] = np.random.rand(sizes[i], sizes[j]) < q1
    
    # Ensure the matrix is symmetric and has no self-loops
    np.fill_diagonal(G, 0)
    G = np.triu(G) + np.triu(G, 1).T

    return G
