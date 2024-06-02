
# import numpy as np

# def test_T2(group):
#     # Simulates a test that categorizes the count of infected individuals in a group
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
#     '''
#     s(np.array): binary string of infection status
#     '''
#     n = len(s)
#     g = 14  # Maximum group size
#     num_tests = 0
#     stages = 0

#     # Step 1: Initialize the array with indices and infection statuses
#     initial_array = np.column_stack((np.arange(n), s, np.zeros(n)))  # Adding a third column for scores

#     # Step 2: Permute and group
#     perm1 = np.random.permutation(initial_array)
#     perm2 = np.random.permutation(initial_array)
#     print(perm1)
#     print()
#     print(perm2)
#     print()
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

#     print(perm1)
#     print()
#     print(perm2)
#     print()

#     for i in range(n):
#         id1 = perm1[i][0]
#         score1 = perm1_dict[id1]
#         score2 = perm2_dict[id1]
#         pair = (score1, score2)
#         print(pair)

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
            
#             if pair in [(3, 2), (2, 3), (2,2)]:
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
#     stages+=1
#     return num_tests, stages
# # Example usage
# s = np.random.randint(0, 2, size=256)  # Mock binary infection status array
# print(s)
# print()
# num_tests, stages = Qtesting2(s)
# print(f"Number of tests: {num_tests}, Number of stages: {stages}")
#----------------------------------------------------------------------------------------------------
# import numpy as np

# def test_T2(group):
#     # Simulates a test that categorizes the count of infected individuals in a group
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
#     '''
#     s(np.array): binary string of infection status
#     '''
#     n = len(s)
#     g = 8  # Maximum group size
#     num_tests = 0
#     stages = 0

#     # Step 1: Initialize the array with indices and infection statuses
#     initial_array = np.column_stack((np.arange(n), s, np.zeros(n)))  # Adding a third column for scores

#     # Step 2: Permute and group
#     perm1 = np.random.permutation(initial_array)
#     perm2 = np.random.permutation(initial_array)
#     print(perm1)
#     print()
#     print(perm2)
#     print()
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

#     print(perm1)
#     print()
#     print(perm2)
#     print()

#     # Store pair values for easy lookup
#     pair_dict = {int(row[0]): (perm1_dict[int(row[0])], perm2_dict[int(row[0])]) for row in perm1}

#     for i in range(n):
#         id1 = perm1[i][0]
#         score1 = perm1_dict[id1]
#         score2 = perm2_dict[id1]
#         pair = (score1, score2)
#         print(pair)

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

#             if pair in [(3, 2)]:
#                 flag = np.any((group_perm1[:, 2] == 4))
#                 if flag:
#                     status = "Uninfected"
#                 else:
#                     # Check for (3, 3) pairs in the group
#                     for individual in group_perm1:
#                         other_id = int(individual[0])
#                         other_pair = pair_dict[other_id]
#                         if other_pair == (3, 3):
#                             status = "Uninfected"
#                             break
#                     else:
#                         if perm1[i][1] == 1:
#                             status = "Infected"
#                         else:
#                             status = "Uninfected"
#                         num_tests += 1
#             elif pair in [(2, 3), (2, 2)]:
#                 flag = np.any((group_perm2[:, 2] == 4))
#                 if flag:
#                     status = "Uninfected"
#                 else:
#                     # Check for (3, 3) pairs in the group
#                     for individual in group_perm2:
#                         other_id = int(individual[0])
#                         other_pair = pair_dict[other_id]
#                         if other_pair == (3, 3):
#                             status = "Uninfected"
#                             break
#                     else:
#                         if perm1[i][1] == 1:
#                             status = "Infected"
#                         else:
#                             status = "Uninfected"
#                         num_tests += 1
#             else:
#                 if perm1[i][1] == 1:
#                     status = "Infected"
#                 else:
#                     status = "Uninfected"
#                 num_tests += 1
#         # Print or store the status for each individual
#         print(f"Individual {id1}: {status}")
#     stages += 1
#     return num_tests, stages

# # Example usage
# s = np.random.randint(0, 2, size=16)  # Mock binary infection status array
# print(s)
# print()
# num_tests, stages = Qtesting2(s)
# print(f"Number of tests: {num_tests}, Number of stages: {stages}")
#----------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

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

def Qtesting2(s, g=8):
    n = len(s)
    num_tests = 0
    stages = 0
    errors = 0

    initial_array = np.column_stack((np.arange(n), s, np.zeros(n)))

    perm1 = np.random.permutation(initial_array)
    perm2 = np.random.permutation(initial_array)
    groups_p1 = [perm1[i:i+g] for i in range(0, n, g)]
    groups_p2 = [perm2[i:i+g] for i in range(0, n, g)]
    
    group_map_p1 = {int(individual[0]): i for i, group in enumerate(groups_p1) for individual in group}
    group_map_p2 = {int(individual[0]): i for i, group in enumerate(groups_p2) for individual in group}

    for group in groups_p1:
        Ct_values = group[:, 1]
        score = test_T2(Ct_values)
        group[:, 2] = score
        num_tests += 1

    for group in groups_p2:
        Ct_values = group[:, 1]
        score = test_T2(Ct_values)
        group[:, 2] = score
        num_tests += 1

    stages += 1

    perm1_dict = {int(row[0]): row[2] for row in perm1}
    perm2_dict = {int(row[0]): row[2] for row in perm2}

    for i in range(n):
        id1 = perm1[i][0]
        score1 = perm1_dict[id1]
        score2 = perm2_dict[id1]
        pair = (score1, score2)

        if pair in [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (2, 0), (3, 0), (4, 0)]:
            status = "Uninfected"
        elif pair in [(4, 4), (4, 1), (1, 4), (4, 2), (2, 4), (4, 3), (3, 4)]:
            status = "Infected"
        elif pair in [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)]:
            if perm1[i][1] == 1:
                status = "Infected"
            else:
                status = "Uninfected"
            num_tests += 1
        elif pair in [(2, 2), (2, 3), (3, 2), (3, 3)]:
            group_idx_p1 = group_map_p1[id1]
            group_idx_p2 = group_map_p2[id1]

            group_perm1 = groups_p1[group_idx_p1]
            group_perm2 = groups_p2[group_idx_p2]
            
            if pair in [(3, 2), (2, 3), (2,2)]:
                flag = np.any((group_perm1[:, 2] == 4))
                
                if flag: 
                    status = "Uninfected"
                
                else:
                    if perm1[i][1] == 1:
                        status = "Infected"
                    else:
                        status = "Uninfected"
                        num_tests += 1
            else: 
                flag = np.any((group_perm2[:, 2] == 4))
                if flag: 
                    status = "Uninfected"
                else:
                    if perm1[i][1] == 1:
                        status = "Infected"
                    else:
                        status = "Uninfected"
                        num_tests += 1

        # Check if the determined status matches the actual status
        if (status == "Infected" and perm1[i][1] != 1) or (status == "Uninfected" and perm1[i][1] != 0):
            errors += 1
    
    stages += 1
    return num_tests, stages, errors

def run_simulations():
    group_sizes = range(2, 17)  # Test different group sizes from 2 to 16
    trials = 100
    n = 264 # Number of individuals in each simulation
    
    avg_tests = []
    avg_errors = []

    for g in group_sizes:
        total_tests = 0
        total_errors = 0
        for _ in range(trials):
            s = np.random.randint(0, 2, size=n)
            num_tests, _, errors = Qtesting2(s, g=g)
            total_tests += num_tests
            total_errors += errors
        avg_tests.append(total_tests / trials)
        avg_errors.append(total_errors / trials)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Group Size')
    ax1.set_ylabel('Average Number of Tests', color='tab:blue')
    ax1.plot(group_sizes, avg_tests, marker='o', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Number of Errors', color='tab:red')
    ax2.plot(group_sizes, avg_errors, marker='x', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title(f'Average Number of Tests and Errors vs Group Size (Test population = {n})' )
    plt.grid(True)
    plt.show()

run_simulations()

