'''
Created on 2022-10-28 11:35:11
@author: caitgrasso

Description: Plot fitness curves of evolutionary run.
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle

fig, ax = plt.subplots(1, 2, sharey=True, figsize = (10,5))

gen = 350
with open(f"afpo_run0_{gen}gens.p", 'rb') as f:
    afpo, rng_state, np_rng_state = pickle.load(f)
afpo.print_best(True)

fitness_data = afpo.fitness_data[:gen,:,0]

max_fits = np.max(fitness_data,axis=1)
best_swarm_fitness = max_fits[gen-1]
ax[0].plot(max_fits,  label="new mutation")
ax[0].set_xlabel('Generations')
ax[0].set_ylabel('Fitness')
ax[0].title.set_text('Evolution Over Time')

#plot histogram of random swarm compositions 
f = open("random_swarms.csv", "r")
data = []
for line in f: 
    split = line.split(",")[1:16]
    lst = []
    for item in split:
        lst.append(float(item))
    data.append(min(lst))

ax[1].hist(data, bins = 'auto', orientation='horizontal', label = "Random Swarms")
ax[1].scatter(0, best_swarm_fitness, label = "Best Evolved", marker = '_', color = "red", s = 200)
ax[1].set_xlabel("Count")
ax[1].title.set_text('Distribution of Random Swarms')

plt.legend()
plt.tight_layout()
plt.savefig(f"random_evolved_comp_{gen}.png")