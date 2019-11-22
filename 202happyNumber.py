def isHappy(n):
    if n < 2:
        return n == 1
    resList = []
    resList.append(n)
    while n>1:
       numSList = [int(i)*int(i) for i in str(n)]
       n=sum(numSList)
       if n in resList:
           return False
       else:
           resList.append(n)
    return True
print(isHappy(500156))

class Solution:
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        seen = {}
        while True:
            if str(n) in seen:
                return n == 1
            seen[str(n)] = True
            newN = 0
            while n > 0:
                n, mod = divmod(n, 10)
                newN += mod ** 2
            n = newN
        return False