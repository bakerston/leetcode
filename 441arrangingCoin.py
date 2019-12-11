def arrangeCoins( n):
    if n<0:
        return False
    if n<100:
        throw=0
        while throw*(throw+1)/2<=n:
            throw=throw+1
        return throw-1
    estThrow=int(pow(2*n,0.5)+1)
    while estThrow*(estThrow+1)/2>n:
        estThrow-=1
    return estThrow
print(arrangeCoins(2))