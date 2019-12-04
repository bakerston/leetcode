def findDisappearedNumbers(nums):
    for pos in range(len(nums)):
        newPos=abs(nums[pos])-1
        nums[newPos]=-1*abs(nums[newPos])
    clist=[i+1 for i in range(len(nums)) if nums[i]>0]
    return clist


print(findDisappearedNumbers([4,3,2,7,8,2,3,1]
))