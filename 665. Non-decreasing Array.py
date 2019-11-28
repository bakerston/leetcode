def checkPossibility(nums):
    moveNum=0
    i=0
    for i in range(len(nums)-1):
        if nums[i]>nums[i+1]:
            if 0<i<len(nums)-2:
                if nums[i]>nums[i+2] and nums[i+1]<nums[i-1]:
                    return False
            moveNum+=1
    return moveNum<=1
#def checkPossPure(nums):
#    for i in range(len(nums) - 1):
#        if nums[i]>nums[i+1]:
#            return False
#    return True



print(checkPossibility( [1,2,3,10,1,2]))