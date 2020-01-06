import collections
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        oribin='{0:032b}'.format(n)
        oList=list(oribin)
        count=0
        for pos in range(len(oList)):
            if oList[pos]=='1':
                count+=1
        return count

    return bin(n).count("1")