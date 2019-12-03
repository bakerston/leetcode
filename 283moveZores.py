def moveZeroes( nums):
    aLen=len(nums)
    pos=0
    while pos<aLen:
        if nums[pos]==0:
            nums.remove(0)
            nums.append(0)
            aLen-=1
        else:
            pos+=1
    return nums

print(moveZeroes([1,0,2,0,3]))
