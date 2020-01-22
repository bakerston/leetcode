def searchInsert( nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    pos = 0
    Found = False
    while pos < len(nums) and not Found:
        if target> nums[pos]:
            pos += 1
        else:
            Found=True
            return pos
    if pos == len(nums) - 1 and not Found:
        return len(nums)

print(searchInsert([1,3,5,6], 5)
)