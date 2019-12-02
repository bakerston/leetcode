def countPrimeSetBits(L, R):
    if L>R:
        return False
    primeList=[2,3,5,7,11,13]
    primeNum=0
    for num in range(L,R+1):
        if bin(num).count('1') in primeList:
            primeNum+=1
    return primeNum
print(countPrimeSetBits(6,10))