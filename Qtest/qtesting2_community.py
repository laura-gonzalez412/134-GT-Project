# import numpy as np

# def test_T2(group):
#     """
#     Simulates a test that categorizes the count of infected individuals in a group.
#     """
#     infected_count = np.sum(group)
#     if infected_count == 0:
#         return 0
#     elif 1 <= infected_count < 2:
#         return 1
#     elif 2 <= infected_count < 4:
#         return 2
#     elif 4 <= infected_count < 8:
#         return 3
#     else:
#         return 4

# def Qtesting2(s):
#     """
#     Perform adaptive group testing using T2 tests on the entire population.
    
#     Parameters:
#     s (np.array): Binary array of infection status.
    
#     Returns:
#     num_tests (int): Total number of tests used.
#     stages (int): Total number of stages used.
#     """
#     n = len(s)
#     g = 8  # Maximum group size
#     num_tests = 0
#     stages = 0

#     # Step 1: Initialize the array with indices and infection statuses
#     initial_array = np.column_stack((np.arange(n), s, np.zeros(n)))  # Adding a third column for scores

#     # Step 2: Permute and group
#     perm1 = np.random.permutation(initial_array)
#     perm2 = np.random.permutation(initial_array)
#     groups_p1 = [perm1[i:i+g] for i in range(0, n, g)]
#     groups_p2 = [perm2[i:i+g] for i in range(0, n, g)]
    
#     # Create group mappings
#     group_map_p1 = {int(individual[0]): i for i, group in enumerate(groups_p1) for individual in group}
#     group_map_p2 = {int(individual[0]): i for i, group in enumerate(groups_p2) for individual in group}

#     # Step 3: First stage testing
#     for group in groups_p1:
#         Ct_values = group[:, 1]  # Infection statuses Ct is a 1D array with the infection statuses of the individuals
#         score = test_T2(Ct_values)
#         group[:, 2] = score
#         num_tests += 1

#     for group in groups_p2:
#         Ct_values = group[:, 1]
#         score = test_T2(Ct_values)
#         group[:, 2] = score
#         num_tests += 1

#     stages += 1

#     # Step 4: Cross-reference and classify
#     perm1_dict = {int(row[0]): row[2] for row in perm1}  # Dictionary of scores from perm1
#     perm2_dict = {int(row[0]): row[2] for row in perm2}  # Dictionary of scores from perm2

#     for i in range(n):
#         id1 = perm1[i][0]
#         score1 = perm1_dict[id1]
#         score2 = perm2_dict[id1]
#         pair = (score1, score2)

#         if pair in [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (2, 0), (3, 0), (4, 0)]:
#             status = "Uninfected"
#         elif pair in [(4, 4), (4, 1), (1, 4), (4, 2), (2, 4), (4, 3), (3, 4)]:
#             status = "Infected"
#         elif pair in [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)]:
#             if perm1[i][1] == 1:
#                 status = "Infected"
#             else:
#                 status = "Uninfected"
#             num_tests += 1
#         elif pair in [(2, 2), (2, 3), (3, 2), (3, 3)]:
#             group_idx_p1 = group_map_p1[id1]
#             group_idx_p2 = group_map_p2[id1]

#             group_perm1 = groups_p1[group_idx_p1]
#             group_perm2 = groups_p2[group_idx_p2]
            
#             if pair in [(3, 2), (2, 3), (2, 2)]:
#                 flag = np.any((group_perm1[:, 2] == 4))
#                 if flag: 
#                     status = "Uninfected"
#                 else:
#                     if perm1[i][1] == 1:
#                         status = "Infected"
#                     else:
#                         status = "Uninfected"
#                         num_tests += 1
#             else: 
#                 flag = np.any((group_perm2[:, 2] == 4))
#                 if flag: 
#                     status = "Uninfected"
#                 else:
#                     if perm1[i][1] == 1:
#                         status = "Infected"
#                     else:
#                         status = "Uninfected"
#                         num_tests += 1
    
#         # Print or store the status for each individual
#         print(f"Individual {id1}: {status}")
#     stages += 1
#     return num_tests, stages

# def Qtesting2_comm_aware(s, communities):
#     """
#     Perform adaptive group testing using T2 tests considering community structure.
    
#     Parameters:
#     s (numpy array): Infection status array (1 for infected, 0 for not infected).
#     communities (list of lists): Each sublist represents a community with indices of individuals in that community.
    
#     Returns:
#     total_tests (int): Total number of tests used.
#     total_stages (int): Total number of stages used.
#     """
#     total_tests = 0
#     total_stages = 0
    
#     for community in communities:
#         sample_size = min(5, len(community))  # Use a small sample size for initial testing
#         representative_sample = np.random.choice(community, sample_size, replace=False)  # Randomly select representatives
        
#         # Perform initial test on the combined sample
#         initial_group = s[representative_sample]
#         initial_tests = 1
#         initial_stages = 1
#         initial_infected_count = test_T2(initial_group)
        
#         total_tests += initial_tests
#         total_stages = max(total_stages, initial_stages)
        
#         # If initial tests indicate infections, perform detailed testing within the community
#         if initial_infected_count > 0:
#             community_s = s[community]
#             community_tests, community_stages = Qtesting2(community_s)
#             total_tests += community_tests
#             total_stages = max(total_stages, community_stages)
    
#     return total_tests, total_stages
import numpy as np

def test_T2(group):
    infected_count = np.sum(group)
    if infected_count == 0:
        return 0
    elif 1 <= infected_count < 2:
        return 1
    elif 2 <= infected_count < 4:
        return 2
    elif 4 <= infected_count < 8:
        return 3
    else:
        return 4

def Qtesting2_comm_aware(s, communities):
    num_tests = 0
    stages = 0

    for community in communities:
        initial_tests = 1
        initial_stages = 1
        initial_test_range = test_T2(s[community])

        num_tests += initial_tests
        stages = max(stages, initial_stages)

        if initial_test_range > 0:
            if initial_test_range == 4:
                num_tests += 0
            else:
                community_group_test = test_T2(s[community])
                num_tests += 1
                stages = max(stages, 2)

                if community_group_test > 0:
                    community_tests = len(community)
                    num_tests += community_tests
                    stages = max(stages, 3)
                else:
                    pass
        else:
            stages = max(stages, 2)

    return num_tests, stages

def Qtesting2_comm_aware(s, communities):
    """
    Perform adaptive group testing using T2 tests considering community structure.
    
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
        initial_infected_count = test_T2(initial_group)
        
        total_tests += initial_tests
        total_stages = max(total_stages, initial_stages)
        
        # If initial tests indicate infections, perform detailed testing within the community
        if initial_infected_count > 0:
            community_s = s[community]
            community_tests, community_stages = Qtesting2(community_s)
            total_tests += community_tests
            total_stages = max(total_stages, community_stages)
    
    return total_tests, total_stages

# Example usage
s = np.array([0, 1, 0, 0, 1, 1, 0, 0, 1, 0])
communities = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
total_tests, total_stages = Qtesting2_comm_aware(s, communities)
print(f"Total tests: {total_tests}, Total stages: {total_stages}")
