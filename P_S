import random
import numpy
def factorial(num):
    a = 1
    while(num):
        a *=num
        num-=1
    return a

def combination(n,x):
    result = factorial(n)/(factorial(x)*factorial(n-x))
    return result

def addseq(start,end,step):
    res = []
    i=start
    while i<end:
        res.append(i)
        i+=step
    return res
    
def mulseq(start,end,step):
    res = []
    i=start
    while i<end+1:
        res.append(i)
        i*=step
    return res

def mean(list):
    res = 0
    n = len(list)
    for i in range(n):
        res +=list[i]

    res /= n
    return res
def val(list):
    mean_list = mean(list)
    val = 0
    for i in list:
        val += (list[i]-mean_list)**2
    return val
def std(list):
    val = val(list)
    res = val**0.5
    return res

def createdata(list , len):
    for i in range(0,len):
        list.append(random.randint(0,10))
    return list
a_list = []
list_len = 10
a_list = createdata(a_list,list_len)

b_list = []
b_list = createdata(b_list,list_len)
print(a_list,b_list)

mean_from_a_list = mean(a_list)
mean_from_b_list = mean(b_list)

sm = 0

for i in range(0,list_len):
    sm+=(a_list[i] - mean_from_a_list)*(b_list[i] - mean_from_b_list)

cov = sm/(list_len-1)

print(cov)