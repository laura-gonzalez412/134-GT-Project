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
    Simulate the T1 test which returns the number of infected individuals in the group.
    """
    return np.sum(group)

def Qtesting1_comm_aware(s, communities):
    '''
    s(np.array): binary string of infection status
    communities(list): the community information
    '''
    num_tests = 0
    stages = 0

    for community in communities:
        sample_size = min(5, len(community))  # Use a small sample size for initial testing
        representative_sample = np.random.choice(community, sample_size, replace=False)  # Randomly select representatives
        
        # Perform initial test
        initial_group = s[representative_sample]
        initial_tests = 1
        initial_stages = 1
        initial_infected_count = test_T1(initial_group)

        num_tests += initial_tests
        stages = max(stages, initial_stages)

        if initial_infected_count > 0:
            # If initial test is positive, test each member individually
            community_tests = len(community)
            num_tests += community_tests
            stages = max(stages, 2)  # One stage for initial test, one for individual tests

        else:
            # If initial test is negative, test the entire community as a group
            community_group_test = test_T1(s[community])
            num_tests += 1
            stages = max(stages, 2)

            if community_group_test > 0:
                # If the group test is positive, test each member individually
                community_tests = len(community)
                num_tests += community_tests
                stages = max(stages, 3)  # One stage for initial test, one for group test, one for individual tests
            else:
                # If the group test is negative, all members are uninfected
                break

    return num_tests, stages

# Example usage
s = np.array([0, 1, 0, 0, 1, 1, 0, 0, 1, 0])
communities = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
total_tests, total_stages = Qtesting1_comm_aware(s, communities)
print(f"Total tests: {total_tests}, Total stages: {total_stages}")


