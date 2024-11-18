import numpy as np
import matplotlib.pyplot as plt
import csv

# divide into 2 arrays as this is a jagged csv
rows_with_final_epilength = []
rows_without_final_epilength = []

# read the csv file
with open("results/parameter_sweep_log.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        # check the number of elements in each row
        if len(row) == 9:
            rows_without_final_epilength.append(row)
        elif len(row) == 10:
            rows_with_final_epilength.append(row)
rows_with_final_epilength = np.array(rows_with_final_epilength)
rows_without_final_epilength = np.array(rows_without_final_epilength)

final_epilength = rows_with_final_epilength[:,-1]
total_array = np.concatenate((rows_without_final_epilength, rows_with_final_epilength[:,:-1]), axis=0)

reset_rates = (total_array[:,0]).astype(np.float64)
integrated_regret = (total_array[:,7]).astype(np.float64)
learning_end_epinum = (total_array[:,8]).astype(np.float64)

plt.figure()
plt.plot(reset_rates, integrated_regret, '.')
plt.xlabel('reset rate')
plt.ylabel('average integrated regret')
plt.yscale('log')
# at small t:
plt.xlim(0,0.005)
plt.ylim(3.4e5,4.6e5)
# plt.savefig('figs/regret_vs_reset_2.png')
plt.show()

plt.figure()
plt.plot(reset_rates, learning_end_epinum, '.')
plt.xlabel('reset rate')
plt.ylabel('average episode number at which learning ends')
# plt.yscale('log')
# plt.xscale('log')
# plt.savefig('figs/regret_vs_reset_2.png')
plt.show()


