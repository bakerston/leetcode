from functools import reduce
def checkPerfectNumber(num):
    return 2*num==sum(set(reduce(list.__add__, ([i, num // i] for i in range(1, int(pow(num, 0.5) + 1)) if num % i == 0))))

print(checkPerfectNumber(28))