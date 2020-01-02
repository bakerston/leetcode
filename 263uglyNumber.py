def isUgly(num):
    """
    :type num: int
    :rtype: bool
    """
    if num == 1:
        return True
    while num % 5 == 0:
        num = num / 5
    while num % 3 == 0:
        num = num / 3
    while num % 2 == 0:
        num = num / 2
    return num == 1
print(isUgly(2))

if num <= 0:
            return False
        for i in [2, 3, 5]:
            while num%i == 0:
                num /= i
        return num == 1