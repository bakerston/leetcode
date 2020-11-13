#1342. Number of Steps to Reduce a Number to Zero
"""def numberOfSteps(num):
    ans=0
    while num!=0:
        if num%2==1:
            num-=1
        else:
            num/=2
        ans+=1
    return ans
print(numberOfSteps(8))"""
#1281. Subtract the Product and Sum of Digits of an Integer
"""from functools import reduce

def subtractProductAndSum(n):
    res=list(map(lambda x:int(x),list(str(n))))
    su=sum(res)
    mu=reduce(lambda x,y:x*y,res)
    return mu-su
print(subtractProductAndSum(4421))"""

def subtractProductAndSum(n):
    alist=list(str(n))
    mu=1
    su=0
    for i in range(len(alist)):
        mu*=int(alist[i])
        su+=int(alist[i])
    return mu-su
print(subtractProductAndSum(4421))