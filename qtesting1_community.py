import numpy as np

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
    def binary_splitting_T1(group):
        """
        Perform binary splitting using T1 tests on a given group.
        
        Parameters:
        group (list): Indices of individuals in the group.
        
        Returns:
        tests (int): Number of tests used.
        stages (int): Number of stages used.
        """
        if len(group) == 1:
            return (1, 1)  # One test, one stage for a single individual
        
        tests = 0
        stages = 0
        
        # Initialize groups for splitting
        groups = [group]
        
        while len(groups) > 0:
            new_groups = []
            stages += 1
            
            for g in groups:
                if len(g) == 1:
                    tests += 1
                    continue
                
                # Split the group into two halves
                mid = len(g) // 2
                left_group = g[:mid]
                right_group = g[mid:]
                
                # Perform T1 test
                left_infected = sum(s[left_group])
                right_infected = sum(s[right_group])
                
                tests += 1
                
                if left_infected > 0:
                    new_groups.append(left_group)
                
                if right_infected > 0:
                    new_groups.append(right_group)
            
            groups = new_groups
        
        return (tests, stages)
    
    total_tests = 0
    total_stages = 0
    
    # Perform initial testing on representative samples from each community
    for community in communities:
        representative_sample = community[:min(5, len(community))]  # Take up to 5 representatives
        initial_tests, initial_stages = binary_splitting_T1(representative_sample)
        total_tests += initial_tests
        total_stages = max(total_stages, initial_stages)
        
        # If initial tests indicate infections, perform detailed testing within the community
        if sum(s[representative_sample]) > 0:
            community_tests, community_stages = binary_splitting_T1(community)
            total_tests += community_tests
            total_stages = max(total_stages, community_stages)
    
    return total_tests, total_stages

# Example usage
s = np.array([0, 1, 0, 0, 1, 1, 0, 0, 1, 0])
communities = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
total_tests, total_stages = Qtesting1_comm_aware(s, communities)
print(f"Total tests: {total_tests}, Total stages: {total_stages}")