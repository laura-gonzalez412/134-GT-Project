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



# Example usage
s = np.random.randint(0, 2, size=264)  # Mock binary infection status array
print(s)
print()
st = []
num_tests, stages, st = binary_splitting(s)
print(f"Number of tests: {num_tests}, Number of stages: {stages}")
print(st)