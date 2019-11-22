def findNthDigit(n):
    if n <= 0:
        return False
    pos=0
    while n>pos*pow(10,pos)-(pow(10,pos)-1)/9:
        pos+=1
    newPos=pos-1
    nums,res=divmod((n-newPos*pow(10,newPos)+(pow(10,newPos)-1)/9),pos)
    if int(res)==0:
        targetNum=pow(10,newPos)-1+int(nums)
     #   print(targetNum)
        finList=[int(i) for i in str(targetNum)]
     #   print(finList)
        return finList[-1]
    else:
        targetNum=pow(10,newPos)+int(nums)
    #    print(targetNum)
        finList=[int(i) for i in str(targetNum)]
    #    print(finList)
        return finList[int(res)-1]
print(findNthDigit(2000))


def findNthDigit(self, n):
    n -= 1
    for digits in range(1, 11):
        first = 10**(digits - 1)
        if n < 9 * first * digits:
            return int(str(first + n/digits)[n%digits])
        n -= 9 * first * digits