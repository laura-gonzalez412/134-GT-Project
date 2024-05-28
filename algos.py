import numpy as np
import random


# binary spliting
def binary_splitting_round(s):
    # s: np.array the infectious status & test status (i.e a 2D array)
    num = 0
    flag = sum(s[:,0])>0 # true if there is at least one infected
    assert flag
    stages = 0
    
    #base case of recursion
    if len(s[:,0])==1: #if there is only one person left 
        s[0,1] = s[0,0]
        #else if s[1,1] then it stays [1,1] to represent infected 
        return num,s,stages
    
    #split up the set into two halves
    B1, B2 = np.array_split(s.copy(), 2,axis=0)
    
    #test to see if B1 has at least one infected 
    flag = sum(B1[:,0])>0
    num+=1 #tested one more so increment
    stages += 1
    
    
    if flag: #if true reiterate the process 
        n,stmp,stage = binary_splitting_round(B1)
        s[:len(B1),1] = stmp[:,1]
        
    else: #nobody infected in B1 so they must be in B2
        s[:len(B1),1] = 0 #set all status' of B1 to zero since not infected
        n,stmp,stage = binary_splitting_round(B2) #recursively call binary split on B2
        s[len(B1):,1] = stmp[:,1]
    num += n
    stages += stage
    return num,s,stages 

def binary_splitting(s):
    # modified bs
    # s: 1-d array the infectious status
    st = np.zeros((len(s),2)) #create a 2D array with |s| rows and 2 columns
    st[:,0] = s #set the first column to the infections values (remember that the tests need to confirm these)
    st[:,1] = np.nan #set the second column to the outcomes of the tests for each individual 
    nums = 0
    count = sum(np.isnan(st[:,1])) #count is equal to the number of unconfirmed infection statuses initiall set to |s|
    stages = 0
    
    #the following code will iterate until all people infection status' are confimred 
    # the undetermined people
    while count!=0:
        mask = np.isnan(st[:,1])
        # mask is a boolean array indicating which rows have np.nan in the second column. 
        # np.isnan(st[:,1]) generates a boolean array where each element is True if the corresponding element in 
        # st[:,1] is np.nan and False otherwise.
        
        flag = sum(st[mask,0]>0)>0 #simply tests (pool) if there are any infected people amongst the undetermined ones
        nums += 1
        stages+=1
        if not flag:
            st[mask,1] = 0 #set all undetermined ones to not infected
            
        else:#at lease one infectious individual
            n,stmp,stage = binary_splitting_round(st[mask,:]) #send all undetermined in 2D array to funct.
            # n: Additional number of operations performed.
            # stmp: Temporary array with updated statuses of only the undetermined (masked) elements.
            # stage: Additional number of stages performed.             
            st[mask,1] = stmp[:,1]
            nums += n
            stages += stage
        count = sum(np.isnan(st[:,1]))
        
    assert sum(st[:,0]!=st[:,1])==0 #tests to make sure columns match as they should
    return nums,stages, st[:,1]

# diag----------------------------------------------------
def diagalg_iter(s):
    # s(np.array): binary string of infection status
    k = int(np.log2(len(s)))
    l = int(2**(k-1))
    lp = 0
    p = np.zeros(k+1)
    group = dict()
    num = np.ones(k+1,dtype=np.int32)
    for i in range(k):
        p[i] = sum(s[lp:lp+l])>0
        group[i] = s[lp:lp+l]
        num[i] = l
        lp+=l
        l = l//2

    p[-1] = s[-1]
    group[k] = np.array([s[-1]])
    # p(array): pattern
    # group(dict): indicate the group information
    # num(array): the group size
    return p.astype(np.int32), group,num


def diag_splitting(s):
    # s(np.array): binary string of infection status
    num_tests = 0
    stages = 0
    pattern, group, nums = diagalg_iter(s)
    stages +=1
    num_tests += len(pattern)
    indices = np.where(pattern == 1)[0]
    flag = 0
    for i in indices:
        if nums[i]>1:
            num_test,stage = diag_splitting(group[i])
            num_tests += num_test
            if not flag:
                stages+=stage
                flag = 1
    return num_tests,stages

#----------------------------------Edited Code Below---------------------------------------------------
def get_infected_range(infections):
    """
    Determine the range of infected individuals in a group based on predefined categories.

    Parameters:
    infections (np.array): Array of binary values where 1 indicates infection and 0 indicates no infection.

    Returns:
    tuple: A tuple representing the range of infected individuals (min, max).
    """
    count = np.sum(infections)  # Count the number of infected individuals
    
    if count == 0:
        return (0, 0)  # No one is infected
    elif count == 1:
        return (1, 1)  # Exactly one person is infected
    elif count < 2:
        return (1, 2)  # 1 to less than 2, not typically used unless fractional infections are considered
    elif count < 4:
        return (2, 4)  # 2 to less than 4
    elif count < 8:
        return (4, 8)  # 4 to less than 8
    else:
        return (8, np.inf)  # 8 or more

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
    num_tests = 0
    stages = 0
    ###################################################
    def recursive_test(indices):
        nonlocal num_tests, stages
        
        #base case where we have gone through all people in the set dont need to do anything just return 
        if len(indices) == 0:
            return
        
        #begin new round of testing
        num_tests += 1
        stages += 1
        group = s[indices]
        
        #call test_T2 which magically gies us the estimated number of people who are infected
        category = test_T2(group)

        if category == 0:
            return  # All individuals are negative and the previous recursive call has accounted for this test
                    # no need to return num_tests or stages
        elif category == 4:
            # Since infection count is large split the group into smaller subgroups for further testing
            # similar to Qtesting1
            mid = len(indices) // 2
            recursive_test(indices[:mid])
            recursive_test(indices[mid:])
            
        else: #here we can use the knowledge of the ranges to tell us how we can run our next tests
            # Handle other categories by additional tests based on estimated infections
            lower_bound = 2 ** (category - 1)
            upper_bound = 2 ** category
            # Category 1: lower_bound = 1, upper_bound = 2
            # Category 2: lower_bound = 2, upper_bound = 4
            # Category 3: lower_bound = 4, upper_bound = 8
            
            estimated_infected = (lower_bound + upper_bound) // 2 #gets the average of infected in that range
            infected_indices = np.random.choice(indices, min(estimated_infected, len(indices)), replace=False)
            recursive_test(np.setdiff1d(indices, infected_indices))

    recursive_test(np.arange(len(s)))
    return num_tests, stages
    
def Qtesting1_comm_aware(s,communities):
    '''
    s(np.array): binary string of infection status
    communities(list): the community information
    '''
    num_tests = 0
    stages = 0
    ###################################################
    '''your code here'''
    '''
    s(np.array): binary string of infection status
    communities(list): the community information
    Apply Qtesting1 logic but initiate within communities.
    '''
    ###################################################

    return num_tests,stages

def Qtesting2_comm_aware(s,communities):
    '''
    s(np.array): binary string of infection status
    communities(list): the community information
    '''
    num_tests = 0
    stages = 0
    ###################################################
    '''your code here'''
    '''
    s(np.array): binary string of infection status
    communities(list): the community information
    Implement an adaptive algorithm using range-based tests (T2) that are aware of community structures.
    '''
    if len(s) == 0:
        return num_tests, stages

    # Iterate over each community
    for community in communities:
        if len(community) > 0:
            # Extract the infection statuses for the current community
            community_statuses = s[community]
            
            # Perform an initial test on the community
            infected_range = get_infected_range_estimate(community_statuses)
            num_tests += 1  # Count this initial community test
            stages += 1  # Each community test is considered a new stage if done sequentially

            if infected_range[1] == 0:
                continue  # No further testing needed if the upper bound is 0
            elif infected_range[0] == len(community_statuses):
                continue  # All are infected, no further tests needed if individual status not required

            # If the exact status within the community is unclear, apply further tests
            mid = len(community_statuses) // 2
            num_tests_left, stages_left = Qtesting2_comm_aware(community_statuses[:mid], [range(mid)])
            num_tests_right, stages_right = Qtesting2_comm_aware(community_statuses[mid:], [range(mid, len(community_statuses))])
            
            num_tests += num_tests_left + num_tests_right
            stages += max(stages_left, stages_right)  # Assuming parallel testing within the community

    ###################################################



    return num_tests,stages