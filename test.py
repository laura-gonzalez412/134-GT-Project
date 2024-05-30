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

# def determine_optimal_group_size(n):
#     # Placeholder function, ideally this would determine the optimal group size based on some criteria
#     return max(1, n // 4)  # Example heuristic: group size is a quarter of n

# def Qtesting2(s):
#     '''
#     s(np.array): binary string of infection status
#     '''
#     n = len(s)
#     g = determine_optimal_group_size(n)  # Dynamically determine optimal group size
#     num_tests = 0
#     stages = 0

#     # Step 1: Initialize the array with indices and infection statuses
#     initial_array = np.column_stack((np.arange(n), s))
    
#     # Step 2: Permute and group
#     perm1 = np.random.permutation(initial_array)
#     perm2 = np.random.permutation(initial_array)
    
#     print(perm1)
#     print()
#     print(perm2)
#     print()
#     groups_p1 = [perm1[i:i+g] for i in range(0, n, g)]
#     groups_p2 = [perm2[i:i+g] for i in range(0, n, g)]
    
#     # Step 3: First stage testing
#     Sp1 = []
#     Sp2 = []
#     for group in groups_p1:
#         Ct_values = group[:, 1]  # Infection statuses
#         Sp1.append(test_T2(Ct_values))
#         num_tests += 1
#     for group in groups_p2:
#         Ct_values = group[:, 1]
#         Sp2.append(test_T2(Ct_values))
#         num_tests += 1
    
#     stages += 1
    
#     # Step 4: Track individuals with identifiers
#     first_perm_scores = np.column_stack((perm1[:, 0], [Sp1[i//g] for i in range(n)]))
#     second_perm_scores = np.column_stack((perm2[:, 0], [Sp2[i//g] for i in range(n)]))
    
#     # Step 5: Cross-reference and classify
#     for i in range(n):
#         id1 = int(first_perm_scores[i][0])
#         score1 = int(first_perm_scores[i][1])
#         score2 = int(second_perm_scores[second_perm_scores[:, 0] == id1][0][1])
#         pair = (score1, score2)
        
#         # Based on the pair, classify the individuals (simplified example)
#         # Here you would implement the classification logic based on the score pairs

#     return num_tests, stages

# # Example usage
# s = np.random.randint(0, 2, size=16)  # Mock binary infection status array
# print(s)
# num_tests, stages = Qtesting2(s)
# print(f"Number of tests: {num_tests}, Number of stages: {stages}")


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

def determine_optimal_group_size(n):
    # Placeholder function, ideally this would determine the optimal group size based on some criteria
    return max(1, n // 4)  # Example heuristic: group size is a quarter of n

def Qtesting2(s):
    '''
    s(np.array): binary string of infection status
    '''
    n = len(s)
    g = 8 #because at most 
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
    
    # Step 3: First stage testing
    for group in groups_p1:
        Ct_values = group[:, 1]  # Infection statuses Ct is an 1D array with the infection statuses of the individuals
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
    # Example classification logic based on score pairs
    for i in range(n):
        
        id1 = perm1[i][0]
        score1 = perm1_dict[id1]
        score2 = perm2_dict[id1]
        pair = (score1, score2)
        print(pair)
        
        # implement the classification logic based on the score pairs
        
        #if they got a zero in any then it means they are not infected
        if pair in [(0, 0) , (0,1) , (0,2), (0,3), (0,4), (1,0), (2,0), (3,0), (4,0)]: 
            status = "Uninfected"
        
        #else if they got any 4 in their pairs, they are definitely infected
        elif pair in [(4, 4), (4,1), (1, 4), (4,2), (2,4), (4,3) , (3,4)]:
            status = "Infected"
        
        #code on what conditions are necessary to test individually (do on higher prob)
        elif pair in [(1,1), (1,2), (1,3), (2,1), (3,1)]:
            #perform individual test
            if perm1[i][1] == 1:
                status = "Infected"
            else:
                status = "Uninfected"
                
        #code on when to back reference the persons group and see if there was someone 
        #who was more infected 
        else: #(2,2), (2,3), (3,2), (3,3)
            
            
        # Print or store the status for each individual
        print(f"Individual {id1}: {status}")

    return num_tests, stages

# # Example usage
# s = np.random.randint(0, 2, size=16)  # Mock binary infection status array
# print(s)
# print()
# num_tests, stages = Qtesting2(s)
# print(f"Number of tests: {num_tests}, Number of stages: {stages}")


# Example usage
s = np.random.randint(0, 2, size=16)  # Mock binary infection status array
print(s)
print()
num_tests, stages = Qtesting2(s)
print(f"Number of tests: {num_tests}, Number of stages: {stages}")

