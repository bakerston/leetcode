def intSqrt(intx):
    if intx<2:
        return intx
    starter=0
    ender=intx
    while ender-starter>1:
        mid=int((starter+ender)/2)
        if mid*mid>intx:
            ender=mid
        elif mid*mid<intx:
            starter=mid
        else:
            return mid
    return starter

print(intSqrt(102))
