#import collections
def findErrorNums(nums):
    """numCounter=collections.Counter(nums)
    keyList=list(numCounter.keys())
    valueList=list(numCounter.values())
    dupNum=keyList[valueList.index(2)]

    aLen=len(nums)
    oriSum=int((aLen+1)*aLen/2)
    reaSum=sum(nums)
    diff=oriSum-reaSum
    missNum=int(dupNum+diff)
    return [dupNum,missNum]"""
    missNum=sum(nums)-sum(set(nums))
    
print(findErrorNums([1,2,7,4,5,6,7]))