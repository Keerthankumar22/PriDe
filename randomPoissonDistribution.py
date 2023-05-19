import random
import math
def randomPoissonNumber(lower_bound, upper_bound, mean):
    for _ in range(1000):
        r = random.random()
        l = math.exp(-mean)
        k = lower_bound
        p = upper_bound
        while True:
            p = p*r
            k += 1
            if p<=l:
                break
        if (k-1)<=upper_bound and (k-1)>=lower_bound:
            return k-1 

def randomPoissonNumber_rand(lower_bound, upper_bound, mean):
    ls = []
    cnt = 50
    for _ in range(1000):
        r = random.random()
        l = math.exp(-mean)
        k = lower_bound
        p = upper_bound
        while True:
            p = p*r
            k += 1
            if p<=l:
                break
        if (k-1)<=upper_bound and (k-1)>=lower_bound:
            ls.append(k-1)
        if len(ls) == cnt:
            break
    return random.choice(ls)