def isPerfectSquare(num):
    if num < 8:
        return num in [1, 4]
    end = num
    start = 1
    while end - start > 1:
        mid = int((end + start) / 2)
        print(end,start,mid)
        if int(mid * mid) == num:
            return True
        elif int(mid * mid) > num:
            end = mid
            start=start
        else:
            start = mid
            end=end
    return False
print(isPerfectSquare(55284235876))