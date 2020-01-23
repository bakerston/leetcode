def isPowerOfTwo( n):

    while n > 2:
        cur=n
        if int(str(cur)[-1]) % 2 != 0:
            return False
        else:
            n = cur / 2
    return True
print(isPowerOfTwo(6))

#
#
#  return True if n & (n-1) == 0 and n != 0 else False
