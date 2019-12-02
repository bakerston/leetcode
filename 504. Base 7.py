def convertToBase7(num):
    factor=""
    if num<0:
        num=abs(num)
        factor="-"
    if num<=6:
        return factor+str(num)
    baseStr=""
    while num>=7:
        mul,res=divmod(num,7)
        baseStr+=str(res)
        num=mul
    baseStr+=str(num)
    baseStr+=factor

    return baseStr[::-1]
print(convertToBase7(-2))