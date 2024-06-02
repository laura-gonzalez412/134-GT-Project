# import numpy as np
# import random


# # binary spliting
# def binary_splitting_round(s):
#     # s: np.array the infectious status & test status (i.e a 2D array)
#     num = 0
#     flag = sum(s[:,0])>0 # true if there is at least one infected
#     assert flag
#     stages = 0
    
#     #base case of recursion
#     if len(s[:,0])==1: #if there is only one person left 
#         s[0,1] = s[0,0]
#         #else if s[1,1] then it stays [1,1] to represent infected 
#         return num,s,stages
    
#     #split up the set into two halves
#     B1, B2 = np.array_split(s.copy(), 2,axis=0)
    
#     #test to see if B1 has at least one infected 
#     flag = sum(B1[:,0])>0
#     num+=1 #tested one more so increment
#     stages += 1
    
    
#     if flag: #if true reiterate the process 
#         n,stmp,stage = binary_splitting_round(B1)
#         s[:len(B1),1] = stmp[:,1]
        
#     else: #nobody infected in B1 so they must be in B2
#         s[:len(B1),1] = 0 #set all status' of B1 to zero since not infected
#         n,stmp,stage = binary_splitting_round(B2) #recursively call binary split on B2
#         s[len(B1):,1] = stmp[:,1]
#     num += n
#     stages += stage
#     return num,s,stages 

# def binary_splitting(s):
#     # modified bs
#     # s: 1-d array the infectious status
#     st = np.zeros((len(s),2)) #create a 2D array with |s| rows and 2 columns
#     st[:,0] = s #set the first column to the infections values (remember that the tests need to confirm these)
#     st[:,1] = np.nan #set the second column to the outcomes of the tests for each individual 
#     nums = 0
#     count = sum(np.isnan(st[:,1])) #count is equal to the number of unconfirmed infection statuses initiall set to |s|
#     stages = 0
    
#     #the following code will iterate until all people infection status' are confimred 
#     # the undetermined people
#     while count!=0:
#         mask = np.isnan(st[:,1])
#         # mask is a boolean array indicating which rows have np.nan in the second column. 
#         # np.isnan(st[:,1]) generates a boolean array where each element is True if the corresponding element in 
#         # st[:,1] is np.nan and False otherwise.
        
#         flag = sum(st[mask,0]>0)>0 #simply tests (pool) if there are any infected people amongst the undetermined ones
#         nums += 1
#         stages+=1
#         if not flag:
#             st[mask,1] = 0 #set all undetermined ones to not infected
            
#         else:#at lease one infectious individual
#             n,stmp,stage = binary_splitting_round(st[mask,:]) #send all undetermined in 2D array to funct.
#             # n: Additional number of operations performed.
#             # stmp: Temporary array with updated statuses of only the undetermined (masked) elements.
#             # stage: Additional number of stages performed.             
#             st[mask,1] = stmp[:,1]
#             nums += n
#             stages += stage
#         count = sum(np.isnan(st[:,1]))
        
#     assert sum(st[:,0]!=st[:,1])==0 #tests to make sure columns match as they should
#     return nums,stages, st[:,1]



# # Example usage
# s = np.random.randint(0, 2, size=264)  # Mock binary infection status array
# print(s)
# print()
# st = []
# num_tests, stages, st = binary_splitting(s)
# print(f"Number of tests: {num_tests}, Number of stages: {stages}")
# print(st)


import numpy as np
import matplotlib.pyplot as plt

def test_T1(group):
    # Simulates a test that returns the exact number of infected in the group
    return np.sum(group)
    

# def Qtesting1(s):
#     num_tests = 0
#     stages = 0

#     def recursive_test(indices):
#         nonlocal num_tests, stages
#         if len(indices) == 0:
#             return

#         num_tests += 1
#         stages += 1
#         group = s[indices]
#         infected_count = test_T1(group)

#         if infected_count == 0 or infected_count == len(indices):
#             return  # No further tests needed

#         # Proportional splitting based on the count of infected individuals
#         if infected_count > 0:
#             # Calculate proportion of infected
#             proportion_infected = infected_count / len(indices)
#             split_point = int(len(indices) * proportion_infected)
            
#             # Ensure split_point is neither 0 nor len(indices)
#             split_point = max(1, min(split_point, len(indices) - 1))

#             left_indices = indices[:split_point]
#             right_indices = indices[split_point:]
#             recursive_test(left_indices)
#             recursive_test(right_indices)

#     recursive_test(np.arange(len(s)))
#     return num_tests, stages

#----------------------------------Edited Code Below---------------------------------------------------

def test_T1(group):
    # Simulates a test that returns the exact number of infected in the group
    return np.sum(group)
    
def Qtesting1(s):
    '''
    s(np.array): binary string of infection status
    '''
    num_tests = 0
    stages = 0
    #######################Edited code below############################
    #this code uses recursion to divide an conquer by spliting up the group until all infection status' are determined
    def recursive_test(indices):
        nonlocal num_tests, stages
        
        #handles the base case where we have no more people to test dont count as a test or stage
        if len(indices) == 0:
            return 
        
        #simply start off by testing using the magical T1 test which tells us how many people are infected
        #if it returns something greater than one than we continue 
        num_tests += 1
        stages += 1
        group = s[indices]
        
        infected_count = test_T1(group)

        #this allows us to take care of two extreme case, where everyone is infected or none 
        #infected. This greatly reduces the amount of tests othewise we would have to keep spliting and 
        #testing until all people are determined. This function lets us stop here 
        if infected_count == 0 or infected_count == len(indices):
            return  # No further action needed as entire group is negative or positive
        
        # Split the group into two halves and test each half recursively
        mid = len(indices) // 2
        left_indices = indices[:mid]
        right_indices = indices[mid:]

        recursive_test(left_indices)
        recursive_test(right_indices)

    recursive_test(np.arange(len(s)))
    ####################################################################
    return num_tests,stages

# Simulation parameters
populations = range(16, 265, 8)  # Population sizes from 16 to 264 in increments of 8
num_simulations = 100  # Number of simulations per population size

# Store the results
average_tests_per_population = []

for population in populations:
    total_tests = 0
    for _ in range(num_simulations):
        # Generate a random infection pattern for each simulation
        s = np.random.randint(0, 2, size=population)
        num_tests, _ = Qtesting1(s)
        total_tests += num_tests
    average_tests = total_tests / num_simulations
    average_tests_per_population.append(average_tests)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(populations, average_tests_per_population, marker='o', linestyle='-')
plt.title('Average Number of Tests by Population Size')
plt.xlabel('Population Size')
plt.ylabel('Average Number of Tests')
plt.grid(True)
plt.show()


