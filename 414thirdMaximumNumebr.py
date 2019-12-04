def thirdMax(nums):
    if len(nums)<2:
        return max(nums)
    g=max(nums)
    try:
        b=max([i for i in nums if i<g])
    except ValueError:
        return g
    try:
        s=max([i for i in nums if i<b])
    except ValueError:
        return b
    return s
print(thirdMax([11,11]))

nums = sorted(list(set(nums)))
        if len(nums)<3:
            return max(nums)
        else:
            return nums[-3]
