def findLengthOfLCIS(nums):
    if len(nums) == 1:
        return 1
    maxLen = 1
    curLen = 1
    for pos in range(len(nums) - 1):
        if nums[pos] < nums[pos + 1]:
            curLen += 1
            maxLen = max(maxLen, curLen)
        else:
            curLen = 1
    return maxLen
print(findLengthOfLCIS(  [2,2,2,2,2]
))