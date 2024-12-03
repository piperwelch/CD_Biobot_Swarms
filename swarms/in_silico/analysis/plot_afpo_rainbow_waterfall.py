from glob import glob
import matplotlib.pyplot as plt
import pickle

GENS = 300 # stop at this generation
FIT_COL = 1  # which column is fitness stored (first column is zero)?
AGE_COL = 0  # which column is age stored?

# get all the data files and sort them from generation 1 to GENS
with open("water_fall.p".format(GENS), 'rb') as f:
    fitness_data = pickle.load(f)

pop_size = 100

line_hist = []
gen_age_fit_dict = {}

for gen in range(GENS+1): #for each generation 

    # gen_age_fit_dict[gen] = {0: 0.0}
    gen_age_fit_dict[gen] = {0: 0}  # MIN FITNESS

    for i in range(pop_size): #find the most fit at each clade 
        this_fit = fitness_data[gen, i, 0]
        this_age =int(fitness_data[gen, i, 1])

        if this_age not in gen_age_fit_dict[gen] or this_fit > gen_age_fit_dict[gen][this_age]: # MIN FITNESS
            gen_age_fit_dict[gen][this_age] = this_fit  # most fit at each age level

    if gen > 0:
        for age in gen_age_fit_dict[gen-1]: #for each clade 
            if age+1 not in gen_age_fit_dict[gen] or gen == GENS:  # extinction #find oldest age 
                
                this_line = []
                n = 0
                while age-n > -1:

                    this_line += [gen_age_fit_dict[gen-1-n][age-n]]
                    n += 1

                pre_fill = [None]*(gen-int(age))
                line_hist += [pre_fill + list(this_line[::-1])]

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

for line in line_hist:
    ax.plot(range(len(line)), line, linewidth=0.8)

plt.xlabel("Age in Generations")
plt.ylabel("Fitness")
plt.savefig("waterfall_plot.png")