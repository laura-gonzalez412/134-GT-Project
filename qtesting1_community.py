import numpy as np
import random

def SBM(N, M, q0, q1):
    """
    Generate the Stochastic Block Model (SBM).
    
    Parameters:
    N (int): Total number of nodes.
    M (int): Number of communities.
    q0 (float): Probability of intra-community connections.
    q1 (float): Probability of inter-community connections.
    
    Returns:
    G (np.array): Adjacency matrix of the generated graph.
    """
    
    # Step 1: Assign nodes to communities
    community_membership = np.random.choice(M, N)
    
    # Step 2: Initialize the adjacency matrix with all zeros
    G = np.zeros((N, N), dtype=int)
    
    # Step 3: Generate edges based on probabilities
    for i in range(N):
        for j in range(i + 1, N):
            if community_membership[i] == community_membership[j]:
                if random.random() < q0:
                    G[i, j] = 1
                    G[j, i] = 1  # For undirected graph
            else:
                if random.random() < q1:
                    G[i, j] = 1
                    G[j, i] = 1  # For undirected graph

    return G

def test_T1(group):
    """
    Simulates the T1 test which returns the number of infected individuals in the group.
    """
    return np.sum(group)

def Qtesting1(s):
    """
    Perform adaptive group testing using T1 tests on the entire population.
    
    Parameters:
    s (np.array): Binary array of infection status.
    
    Returns:
    num_tests (int): Total number of tests used.
    stages (int): Total number of stages used.
    """
    num_tests = 0
    stages = 0

    def recursive_test(indices):
        nonlocal num_tests, stages
        
        if len(indices) == 0:
            return
        
        num_tests += 1
        stages += 1
        group = s[indices]
        
        infected_count = test_T1(group)

        if infected_count == 0 or infected_count == len(indices):
            return
        
        mid = len(indices) // 2
        left_indices = indices[:mid]
        right_indices = indices[mid:]

        recursive_test(left_indices)
        recursive_test(right_indices)

    recursive_test(np.arange(len(s)))
    return num_tests, stages

def Qtesting1_comm_aware(s, communities):
    """
    Perform adaptive group testing using T1 tests considering community structure.
    
    Parameters:
    s (numpy array): Infection status array (1 for infected, 0 for not infected).
    communities (list of lists): Each sublist represents a community with indices of individuals in that community.
    
    Returns:
    total_tests (int): Total number of tests used.
    total_stages (int): Total number of stages used.
    """
    total_tests = 0
    total_stages = 0
    
    for community in communities:
        sample_size = min(5, len(community))  # Use a small sample size for initial testing
        representative_sample = np.random.choice(community, sample_size, replace=False)  # Randomly select representatives
        
        # Perform initial test on the combined sample
        initial_group = s[representative_sample]
        initial_tests = 1
        initial_stages = 1
        initial_infected_count = test_T1(initial_group)
        
        total_tests += initial_tests
        total_stages = max(total_stages, initial_stages)
        
        # If initial tests indicate infections, perform detailed testing within the community
        if initial_infected_count > 0:
            community_s = s[community]
            community_tests, community_stages = Qtesting1(community_s)
            total_tests += community_tests
            total_stages = max(total_stages, community_stages)
    
    return total_tests, total_stages

# Example usage
# Example usage with SBM
N = 10
M = 3
q0 = 0.8
q1 = 0.1

G = SBM(N, M, q0, q1)

# Create the infection status array
infection_rate = 0.2
s = np.random.choice([0, 1], size=N, p=[1 - infection_rate, infection_rate])

# Create the communities list from community membership
communities = [[] for _ in range(M)]
community_membership = np.argmax(G, axis=1) % M  # Derive community membership from adjacency matrix
for i in range(N):
    communities[community_membership[i]].append(i)

# Run the community-aware testing with Qtesting1_comm_aware
print("Qtesting1_comm_aware results:")
total_tests, total_stages = Qtesting1_comm_aware(s, communities)
print(f"Total tests: {total_tests}, Total stages: {total_stages}")