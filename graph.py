import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom , poisson ,nbinom
'''
#이산형 균등 분포

a=1
b=5
list = list(range(a,b+1))
print(list)
p_list = []
for i in list:
    prob = 1/(b-a+1)
    p_list.append(prob)

print(p_list)

plt.bar(list,p_list)
plt.savefig("이산형 균등 분포")
'''

'''
#베르누이 분포

x_list = np.random.binomial(n=1,p=0.5,size = 1000)
plt.hist(x_list)
plt.savefig("베르누이예시")
'''
'''
#이항 분포
n=100
p=0.9
x_list = np.arange(0,n+1)
p_list = []

for x in  x_list:
    prob = binom.pmf(k=x,n=n,p=p)
    p_list.append(prob)
print(p_list)
plt.bar(x_list,p_list)
plt.savefig(f"이항분포예시 n : {n},p : {p}.png")
'''
'''
#포아송 분포
lamb = 70
n = 100
x_list = np.arange(0,n)
p_list = []
for x in x_list:
    prob = poisson.pmf(k=x,mu = lamb)
    p_list.append(prob)
plt.bar(x_list,p_list)
plt.savefig("포아송 분포")
'''
'''
#기하 분포
x_list = np.random.geometric(p=0.2, size = 10000)
plt.hist(x_list)
plt.savefig("기하 분포")
'''

'''
#음이항 분포
r=5
p=0.6
x_list = np.arange(r,r+20)
p_list = []

for x in x_list:
    prob = nbinom.pmf(k=x,n=r,p=p)
    p_list.append(prob)
plt.bar(x_list,p_list)
plt.savefig("음이항 분포")
'''
