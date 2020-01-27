def maxSubArray( nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if len(nums)==1:
        return nums[0]
    maxNum=nums[0]
    add=0
    sum=0
    for pos in range(len(nums)):
        sum=sum+nums[pos]
        if sum>maxNum:
            maxNum=sum
        if sum<0:
            sum=0
    return maxNum

print(maxSubArray([-2,-2,-1]))
