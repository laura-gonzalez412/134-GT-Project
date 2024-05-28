import numpy as np
import random


# binary spliting
def binary_splitting_round(s):
    # s: np.array the infectious status & test status
    num = 0
    flag = sum(s[:,0])>0
    assert flag
    stages = 0
    if len(s[:,0])==1:
        s[0,1] = s[0,0]
        return num,s,stages
    
    B1, B2 = np.array_split(s.copy(), 2,axis=0)
    flag = sum(B1[:,0])>0
    num+=1
    stages += 1
    
    if flag:
        n,stmp,stage = binary_splitting_round(B1)
        s[:len(B1),1] = stmp[:,1]
    else:
        s[:len(B1),1] = 0
        n,stmp,stage = binary_splitting_round(B2)
        s[len(B1):,1] = stmp[:,1]
    num += n
    stages += stage
    return num,s,stages 

def binary_splitting(s):
    # modified bs
    # s: 1-d array the infectious status
    st = np.zeros((len(s),2))
    st[:,0] = s
    st[:,1] = np.nan
    nums = 0
    count = sum(np.isnan(st[:,1]))
    stages = 0
    # the undetermined people
    while count!=0:
        mask = np.isnan(st[:,1])
        flag = sum(st[mask,0]>0)>0
        nums += 1
        stages+=1
        if not flag:
            st[mask,1] = 0
        else:
            n,stmp,stage = binary_splitting_round(st[mask,:])
            st[mask,1] = stmp[:,1]
            nums += n
            stages += stage
        count = sum(np.isnan(st[:,1]))
        
    assert sum(st[:,0]!=st[:,1])==0
    return nums,stages, st[:,1]

# diag
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


def Qtesting1(s):
    '''
    s(np.array): binary string of infection status
    '''
    ###################################################
    '''your code here'''
    ###################################################

    return num_tests,stages

def get_infected_range_estimate(s):
    '''
    Simulate the output of a T2 test by returning a tuple (min, max) representing
    the range of possible infected counts based on the actual infections in s.
    '''
    actual_infected = np.sum(s)
    if actual_infected == 0:
        return (0, 0)
    elif actual_infected <= 2:
        return (1, 2)
    elif actual_infected <= 4:
        return (3, 4)
    elif actual_infected <= 8:
        return (5, 8)
    else:
        return (9, len(s))
    
def Qtesting2(s):
    '''
    s(np.array): binary string of infection status
    '''
    ###################################################
    return num_tests,stages



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
    num_tests = 0
    stages = 0
    for community in communities:
        if len(community) > 0:
            community_tests, community_stages = Qtesting1(s[community])
            num_tests += community_tests
            stages = max(stages, community_stages)  # Parallel testing across communities
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