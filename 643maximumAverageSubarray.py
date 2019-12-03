def findMaxAverage(nums,k):
    if k==len(nums):
        return sum(nums)/k
    aLen=len(nums)
    diff=int(aLen-k)
    maxSum=sum(nums[0:k])
    curSum = sum(nums[0:k])
    for pos in range(diff):
        curSum+=nums[k+pos]-nums[pos]
        if curSum>maxSum:
            maxSum=curSum
    return maxSum/k
print(findMaxAverage([1,12,-5,-6,50,3],2))