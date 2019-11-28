def nthUglyNumber( n):
    if n < 6:
        return 6
    nth = 2
    num = 3
    while nth < n:
        if isUgly(num):
            nth += 1
            if nth==n:
                return num
        num += 1


def isUgly( num):
    if num <= 0:
        return False
    for i in [2, 3, 5]:
        while num % i == 0:
            num /= i
    return num == 1
print(nthUglyNumber(1690))