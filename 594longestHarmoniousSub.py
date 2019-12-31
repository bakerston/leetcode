import collections


def findLHS( nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if len(nums) < 2:
        return 0

    nums.sort()
    minNum = nums[0]
    maxNum = nums[-1]

    if minNum == maxNum:
        return 0

    maxLen = 0
    nums.sort()
    counter = collections.Counter(nums)

  #  dic = dict(counter)
    valueList = list(counter.values())
    keyList = list(counter.keys())
 #   print(keyList)
  #  print(valueList)
    for pos in range(len(keyList) - 1):
        curLen = 0
        if keyList[pos + 1] - keyList[pos] == 1:
            curLen = valueList[pos] + valueList[pos + 1]
            if curLen > maxLen:
                maxLen = curLen
    return maxLen

print(findLHS([-1,0,-1,0,-1,0,-1]))

class Solution(object):
    def findLHS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        count = collections.Counter(nums)
        return max([count[x] + count[x+1] for x in count if count[x+1]] or [0])

    def findLHS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        mp = {}
        for i in nums:
            if i not in mp:
                mp[i] = 1
            else:
                mp[i] += 1

        ln = 0;

        for i in mp:
            if mp.get(i + 1):
                ln = max(ln, mp[i] + mp[i + 1])
        return ln
