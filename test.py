
import numpy as np

def test_T2(group):
    # Simulates a test that categorizes the count of infected individuals in a group
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

def Qtesting2(s):
    '''
    s(np.array): binary string of infection status
    '''
    n = len(s)
    g = 8  # Maximum group size
    num_tests = 0
    stages = 0

    # Step 1: Initialize the array with indices and infection statuses
    initial_array = np.column_stack((np.arange(n), s, np.zeros(n)))  # Adding a third column for scores

    # Step 2: Permute and group
    perm1 = np.random.permutation(initial_array)
    perm2 = np.random.permutation(initial_array)
    print(perm1)
    print()
    print(perm2)
    print()
    groups_p1 = [perm1[i:i+g] for i in range(0, n, g)]
    groups_p2 = [perm2[i:i+g] for i in range(0, n, g)]
    
    # Create group mappings
    group_map_p1 = {int(individual[0]): i for i, group in enumerate(groups_p1) for individual in group}
    group_map_p2 = {int(individual[0]): i for i, group in enumerate(groups_p2) for individual in group}

    # Step 3: First stage testing
    for group in groups_p1:
        Ct_values = group[:, 1]  # Infection statuses Ct is a 1D array with the infection statuses of the individuals
        score = test_T2(Ct_values)
        group[:, 2] = score
        num_tests += 1

    for group in groups_p2:
        Ct_values = group[:, 1]
        score = test_T2(Ct_values)
        group[:, 2] = score
        num_tests += 1

    stages += 1

    # Step 4: Cross-reference and classify
    perm1_dict = {int(row[0]): row[2] for row in perm1}  # Dictionary of scores from perm1
    perm2_dict = {int(row[0]): row[2] for row in perm2}  # Dictionary of scores from perm2

    print(perm1)
    print()
    print(perm2)
    print()

    for i in range(n):
        id1 = perm1[i][0]
        score1 = perm1_dict[id1]
        score2 = perm2_dict[id1]
        pair = (score1, score2)
        print(pair)

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
    
        # Print or store the status for each individual
        print(f"Individual {id1}: {status}")
    stages+=1
    return num_tests, stages
# Example usage
s = np.random.randint(0, 2, size=256)  # Mock binary infection status array
print(s)
print()
num_tests, stages = Qtesting2(s)
print(f"Number of tests: {num_tests}, Number of stages: {stages}")

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
