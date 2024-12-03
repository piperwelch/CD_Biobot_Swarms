'''
Created on 2022-10-26 11:34:57
@author: caitgrasso

Description: Driver script to start evolutionary run of AFPO.
'''
import random
import numpy as np
import os
import pickle

from vxa import VXA
from afpo import AFPO

# Inputs for a single run of AFPO
SEED = 0 # change to run # when running replicates of evolution
GENS = 350
POPSIZE = 100 # popsize needs to be large for AFPO
CHECKPOINT_EVERY = 1 # number of gens to save out state of afpo

# Make vxa once in the beginning for all sims
vxa = VXA(SimTime = 5, EnableCilia=1, EnableCollision=1)
vxa.add_material(RGBA=(255,0,0, 255), E=9e4, Cilia=0.0006, LockZ = 1)
vxa.write('data/base.vxa')

afpo = AFPO(random_seed=SEED, gens=GENS, popsize=POPSIZE, checkpoint_every=CHECKPOINT_EVERY)

best, fitness_data = afpo.run()

best.print(verbose=True)
print(fitness_data)

# pickle out best bot and save csv of fitness data
os.makedirs('results/', exist_ok=True)
filename = 'afpo_seed{}_gens{}_popsize{}'.format(SEED, GENS, POPSIZE)
with open('results/'+filename+'_best.p','wb') as f:
    pickle.dump(best, f)
np.savetxt('results/'+filename+'_fitness_data.csv', fitness_data, delimiter=',')
