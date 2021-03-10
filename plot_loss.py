import matplotlib.pyplot as plt
import numpy as np
import os
import re
import seaborn as sb

if not os.path.exists("figures"):
    os.makedirs("figures")

data = {"noisy_loss":[],
        "clean_loss":[],
        "l1_loss":[],
        "gp_loss":[]}
data_list= open("log1.txt").readlines()
for line in data_list:
    line = re.split(" |,|:|\n",line)
    data["clean_loss"].append(float(line[4]))
    data["noisy_loss"].append(float(line[7]))
    data["l1_loss"].append(float(line[13]))
    data["gp_loss"].append(float(line[15]))

def smooth(x,k=200):
    x_ = [np.mean(x[i:i+k]) for i in range(len(x)-k)]
    return x_
data["clean_loss"] = smooth(data["clean_loss"])
data["noisy_loss"] = smooth(data["noisy_loss"])
data["l1_loss"] = smooth(data["l1_loss"])
data["gp_loss"] = smooth(data["gp_loss"])
plt.figure(figsize=[5,4])
sb.set()
plt.plot(data["clean_loss"][:15000],color='green',linewidth=1.5)
plt.plot(data["noisy_loss"][:15000],color='tomato',linewidth=1.5)
plt.legend(["clean","noisy"])
plt.xlabel("Iteration (x1000)")
plt.ylabel("Values of Loss")
plt.xticks([0,4000,8000,12000],[0,4,8,12])
plt.yticks([-60,-40,-20,0,20])
plt.tight_layout()
plt.savefig("./figures/clean_noisy_dis.pdf")

plt.figure(figsize=[5,4])
plt.plot(smooth(np.array(data["noisy_loss"])-np.array(data["clean_loss"]))[:15000],color='tomato',linewidth=1.5)
plt.xlabel("Iteration (x1000)")
plt.ylabel("Wasserstein Distance")
plt.xticks([0,4000,8000,12000],[0,4,8,12])
plt.yticks([-5.0,0.0,5.0,10.0,15.0])
plt.tight_layout()
plt.savefig("./figures/wasserstein.pdf")

plt.show()
