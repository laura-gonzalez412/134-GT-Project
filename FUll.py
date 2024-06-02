import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pickle

def SBM(N, M, q0, q1):
    community_membership = np.random.choice(M, N)
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            if community_membership[i] == community_membership[j]:
                if random.random() < q0:
                    G[i, j] = 1
                    G[j, i] = 1
            else:
                if random.random() < q1:
                    G[i, j] = 1
                    G[j, i] = 1
    return G, community_membership

def detect_communities(community_membership, M):
    communities = [[] for _ in range(M)]
    for idx, community in enumerate(community_membership):
        communities[community].append(idx)
    return communities

def infect(G, p0, p1, time_steps):
    N = G.shape[0]
    individuals = np.zeros(N)
    for i in range(N):
        if random.random() < p0:
            individuals[i] = 1
    for _ in range(time_steps):
        individuals = infect_step(G, p1, individuals, N)
    return individuals

def infect_step(G, p1, individuals, N):
    individuals_updated = np.copy(individuals)
    for i in range(N):
        if individuals[i] == 1:
            for j in range(N):
                if G[i, j] == 1 and individuals[j] == 0:
                    if random.random() < p1:
                        individuals_updated[j] = 1
    return individuals_updated

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

def iter(N, M, q0, q1, p0, p1, time_steps, num_sims, method, dataset='sbm'):
    Gs = np.zeros((num_sims, N, N))
    Communities = dict()
    Individuals = dict()
    if dataset == 'sbm':
        for i in range(num_sims):
            Gs[i], community_membership = SBM(N, M, q0, q1)
            communities = detect_communities(community_membership, M)
            Communities[i] = communities
            Individuals[i] = infect(Gs[i], p0, p1, time_steps)
    elif dataset == 'iid':
        for i in range(num_sims):
            individuals = np.random.binomial(1, p0, N)
            Communities[i] = [list(range(N))]
            Individuals[i] = individuals

    fraction_ppl = []
    fraction_family = []
    fraction_ppl_in_family = []
    test_counts = []

    for i in range(num_sims):
        G = Gs[i]
        communities = Communities[i]
        individuals = Individuals[i]

        total_infected = np.sum(individuals)
        fraction_ppl.append(total_infected / N)

        infected_communities = sum(np.any(individuals[community]) for community in communities)
        fraction_family.append(infected_communities / M)

        avg_infected_per_community = np.mean([np.sum(individuals[community]) / len(community) for community in communities])
        fraction_ppl_in_family.append(avg_infected_per_community)

        numtests_q2_c, num_stages_q2_c = Qtesting2_comm_aware(individuals.copy(), communities)
        test_counts.append(numtests_q2_c)

    avg_fraction_ppl = np.mean(fraction_ppl)
    avg_fraction_family = np.mean(fraction_family)
    avg_fraction_ppl_in_family = np.mean(fraction_ppl_in_family)
    avg_tests = np.mean(test_counts)

    return avg_fraction_ppl, avg_fraction_family, avg_fraction_ppl_in_family, avg_tests

# Parameters
N = 256
M = 16
time_steps = 2
num_sims = 100
method = 'Q2_Comm_Aware'

# Infection probabilities
infection_probs = [(1, 0), (0.9, 0.1), (0.5, 0.3)]
p0_values = np.linspace(0.01, 0.3, 10)

# Run simulations and collect results
results = {q: {"tests": [], "fraction_infected": [], "percentage_infected_comms": [], "avg_infected_per_comm": []} for q in infection_probs}

for q0, q1 in infection_probs:
    for p0 in p0_values:
        avg_fraction_ppl, avg_fraction_family, avg_fraction_ppl_in_family, avg_tests = iter(N, M, q0, q1, p0, 0.1, time_steps, num_sims, method)
        results[(q0, q1)]["tests"].append(avg_tests)
        results[(q0, q1)]["fraction_infected"].append(avg_fraction_ppl)
        results[(q0, q1)]["percentage_infected_comms"].append(avg_fraction_family)
        results[(q0, q1)]["avg_infected_per_comm"].append(avg_fraction_ppl_in_family)

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

for (q0, q1), data in results.items():
    label = f'q0={q0}, q1={q1}'
    axs[0, 0].plot(p0_values, data["tests"], label=label)
    axs[0, 1].plot(p0_values, data["fraction_infected"], label=label)
    axs[1, 0].plot(p0_values, data["percentage_infected_comms"], label=label)
    axs[1, 1].plot(p0_values, data["avg_infected_per_comm"], label=label)

axs[0, 0].set_xlabel('Infection Probability (p0)')
axs[0, 0].set_ylabel('Average Number of Tests')
axs[0, 0].set_title('Average Number of Tests vs Infection Probability')
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].set_xlabel('Infection Probability (p0)')
axs[0, 1].set_ylabel('Fraction of Infected People')
axs[0, 1].set_title('Fraction of Infected People vs Infection Probability')
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[1, 0].set_xlabel('Infection Probability (p0)')
axs[1, 0].set_ylabel('Percentage of Infected Communities')
axs[1, 0].set_title('Percentage of Infected Communities vs Infection Probability')
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].set_xlabel('Infection Probability (p0)')
axs[1, 1].set_ylabel('Average Percentage of Infected People in Each Community')
axs[1, 1].set_title('Average Percentage of Infected People in Each Community vs Infection Probability')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
