def countPrimes(n):
    if n < 4:
        return n == 2 or n == 3
    primeList = [2]
    number = 3
    for number in range(3,n):
        primeList = appendPrimeList(number, primeList)
    return len(primeList)

def appendPrimeList(n,primeList):
    if not 0 in [(n%i) for i in primeList]:
        primeList.append(n)
    return primeList

print(countPrimes(100))

def countPrimes(self, n):
    if n <= 2:
        return 0
    res = [True] * n
    res[0] = res[1] = False
    for i in xrange(2, n):
        if res[i] == True:
            for j in xrange(2, (n-1)//i+1):
                res[i*j] = False
    return sum(res)