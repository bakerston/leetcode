def findDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    aset = set()
    for i in nums:
        if not i in aset:
            aset.add(i)
        else:
