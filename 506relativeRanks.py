def findRelativeRanks( nums):
    sort = sorted(nums)[::-1]
    rank = ["Gold Medal", "Silver Medal", "Bronze Medal"] + list(map(str, range(4, len(nums) + 1)))

    return map(dict(zip(sort, rank)).get, nums)
print(findRelativeRanks([5,6,4,2,1]))