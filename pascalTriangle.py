#insert
class Solution:
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows == 0:
            return []
        if numRows == 1:
            return [[1]]
        if numRows == 2:
            return [[1], [1, 1]]

        finList = [[] for i in range(numRows)]
        finList[0] = [1]
        finList[1] = [1, 1]

        for pos in range(2, numRows):
            finList[pos] = [1]
            finList[pos].insert(1, 1)
            for newPos in range(1, pos):
                finList[pos].insert(1, finList[pos - 1][newPos - 1] + finList[pos - 1][newPos])
            newPos = newPos + 1
        pos = pos + 1
        return finList
# ==
class Solution:
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows == 0:
            return []
        if numRows == 1:
            return [[1]]
        if numRows == 2:
            return [[1], [1, 1]]

        finList = [[1 for j in range(i + 1)] for i in range(numRows)]

        for pos in range(2, numRows):
            finList[pos][1] = 1
            finList[pos][pos] = 1
            for newPos in range(1, pos):
                finList[pos][newPos] = finList[pos - 1][newPos - 1] + finList[pos - 1][newPos]
        #        newPos=newPos+1
        #    pos=pos+1
        return finList[numRows-1]
#pascalTriangle II
class Solution:
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        if rowIndex == 0:
            return [1]
        if rowIndex == 1:
            return [1, 1]

        finList = [[1 for j in range(i + 1)] for i in range(rowIndex + 1)]

        for pos in range(2, rowIndex + 1):
            finList[pos][1] = 1
            finList[pos][pos] = 1
            for newPos in range(1, pos):
                finList[pos][newPos] = finList[pos - 1][newPos - 1] + finList[pos - 1][newPos]
        #        newPos=newPos+1
        #    pos=pos+1
        return finList[rowIndex]


#leetcode:
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        row = [1]
        for _ in range(rowIndex):
            row = [x + y for x, y in zip([0]+row, row+[0])]
        return row