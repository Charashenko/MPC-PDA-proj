import matplotlib.pyplot as plt
import os
num_of_nns = 5

plt.ylim((-200, 200))
for i in range(1, num_of_nns +1):
    with open(f"rewards/reward_{i}") as f:
        lines = f.readlines()
        ln = range(1, len(lines)+1)
        data = [float(x.strip()) for x in lines]
        plt.plot(ln, data)
plt.show()