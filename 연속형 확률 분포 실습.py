import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm,gamma,expon

'''
#정규분포
mu = 75
s = 3
x_list = np.arange(mu-3*s,mu+3*s,0.01)
p_list = []

for x in x_list:
    prod = norm.pdf(x, loc = mu, scale = s)
    p_list.append(prod)

plt.plot(x_list,p_list)
plt.savefig("정규분포 예제")
plt.close()
y_list = np.random.normal(loc = 75,scale=s, size = 1000)
plt.hist(y_list)
plt.savefig("정규분포 난수")
'''

'''
#감마분포
alpha = 10
beta = 0.01

x_list = np.arange(0, alpha*beta*5, 0.01)
p_list = []
for x in x_list:
    prob = gamma.pdf(x,a=alpha,scale = beta)
    p_list.append(prob)
plt.plot(x_list,p_list)
plt.savefig("감마분포 예제")
plt.close()
y_list = np.random.gamma(2,1,10000)
plt.hist(y_list)
plt.savefig("감마분포 난수")
'''

'''
#지수분포
beta = 1
x_list = np.arange(0,beta*10,0.01)
p_list = []
for x in x_list:
    prob = expon.pdf(x,scale = beta)
    p_list.append(prob)
plt.plot(x_list,p_list)
plt.savefig("지수분포 예제")

y_list = np.random.exponential(scale = 5,size=1000)
plt.hist(y_list)
plt.savefig("지수분포 난수")
'''
