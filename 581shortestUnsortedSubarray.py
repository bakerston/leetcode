def findUnsortedSubarray(nums):
    """if len(nums)==0:
        return 0
    minimum=min(nums)
    maximum=max(nums)
    if len(nums)==1:
        return 0
    if nums[0]==minimum and nums[-1]==maximum:
        return findUnsortedSubarray(nums[1:-1])
    elif nums[0]!=minimum and nums[-1]==maximum:
        return findUnsortedSubarray(nums[:-1])
    elif nums[0]==minimum and nums[-1]!=maximum:
        return findUnsortedSubarray(nums[1:])
    else:
        return len(nums)"""
    is_same = [a == b for a, b in zip(nums, sorted(nums))]
    0 if all(is_same) else len(nums) - is_same.index(False) - is_same[::-1].index(False)
print(findUnsortedSubarray([1,5,7,2,2,3,4]))