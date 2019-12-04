"""import collections
from functools import reduce
def repeatedSubstringPattern(s):
    scounter=collections.Counter(s)
    subTime=min(list(scounter.values()))
    if subTime==1:
        return False
    #set(reduce(list.__add__, ([i, num // i] for i in range(1, int(pow(num, 0.5) + 1)) if num % i == 0)))
    factorList=list(getFactor(subTime))
    factorList1=[i for i in factorList if i!=1]
    for subTime in factorList1:
        if len(s)%subTime==0:
            subLength = int(len(s) / subTime)
            isSub=True
            pos=1
            while pos in range(1, subTime) and isSub:
                if s[0:subLength] != s[pos * subLength:(pos + 1) * subLength]:
                    isSub=False
                else:
                    isSub=True
                    pos+=1
            if pos==subTime:
                return True
    return False


def getFactor(n):
    n=int(n)
    return set(reduce(list.__add__, ([i, n // i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0)))
print(repeatedSubstringPattern("abbababbab"))"""


def repeatedSubstringPattern( s):
    return s in (s + s)[1:-1]
print(repeatedSubstringPattern("cnmcnm"))