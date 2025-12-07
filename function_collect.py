import random
import numpy
def factorial(num):
    '''
    :param num: num! 구하기
    :return:
    '''
    a = 1
    while(num):
        a *=num
        num-=1
    return a

def combination(n,x):
    '''
    :param n: n개 중에
    :param x: x개 고르는 경우의 수 구하기
    :return:
    '''
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
    '''
    리스트 평균 구하기
    '''
    res = 0
    n = len(list)
    for i in range(n):
        res +=list[i]

    res /= n
    return res
def val(list):
    '''
    리스트 분산 구하기
    '''
    n = len(list)
    mean_list = mean(list)
    val = 0
    for i in list:
        val += (list[i]-mean_list)**2
    val /= (n-1)
    return val

def std(list):
    '''
    리스트 편차 구하기
    '''
    val_res = val(list)
    res = val_res**0.5
    return res

def createdata(list , len):
    '''
    0부터 10까지 무작위 숫자 넣기
    '''
    for i in range(0,len):
        list.append(random.randint(0,10))
    return list

def count_freq(data):
    '''
    :param data: 리스트 안에 값들의 빈도 수 구하기
    :return: 각 값을 key로, 빈도 수를 value로 하는 빈도 딕셔너리
    '''
    data_freq = {}
    keys = list(set(data))
    keys.sort()
    for key in keys:
        data_freq[key] = 0
    for v in data:
        data_freq[v] +=1
    return data_freq
def freq2ratio(dic):
    '''
    :param dic: 빈도 딕셔너리를 비율로 변환
    :return: 비율 딕셔너리
    '''
    n = sum(dic.values())
    res = {}
    for key in dic.keys():
        val = dic[key]
        res[key]= val/n
    return res
def pseudo_sample(x0 = 16809,mod = (2**31)-1, seed = 181828, size = 1):
    '''
    유사 난수 생성(0~1 범위)
    :param x0: 초기 x0 값
    :param mod: 난수 주기
    :param seed: 랜덤 시드
    :param size: 난수 개수
    :return: (0~1범위)를 가지는 실수 난수 리스트
    '''
    res = []
    x = (x0 * seed+1)%mod
    u = x / mod
    res.append(u)
    for i in range(1,size):
        x = (x * seed + 1) % mod
        u = x / mod
        res.append(u)
    return res
def sample_from_data(data,seed = 181828,size = 1):
    '''
    data 분포를 따르는 샘플 추출
    :param data: 바탕이 될 분포
    :param seed:
    :param size: 샘플 개수
    :return: 샘플 리스트
    '''
    data_freq = count_freq(data)
    data_ratio = freq2ratio(data_freq)

    sample_list = pseudo_sample(seed = seed,size = size)
    res = []
    for sample in sample_list:
        prob = 0
        for x in data_ratio.keys():
            prob +=data_ratio[x]
            if prob>sample:
                res.append(x)
                break
    return res
