def minMoves(nums):

    minNum = min(nums)
    return sum([i - minNum for i in nums])
print(minMoves([1,2,3]))