import numpy as np

def test_T1(group):
    """
    Simulate the T1 test which returns the number of infected individuals in the group.
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
        
        # Perform initial test
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
s = np.array([0, 1, 0, 0, 1, 1, 0, 0, 1, 0])
communities = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
total_tests, total_stages = Qtesting1_comm_aware(s, communities)
print(f"Total tests: {total_tests}, Total stages: {total_stages}")
