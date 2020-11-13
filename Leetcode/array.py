#1365 How Many Numbers Are Smaller Than the Current Number
"""def smallerNumbersThanCurrent(nums):
    res=sorted(nums)
    adic={}
    for i in range(len(nums)):
        if res[i] not in adic.keys():
            adic[res[i]]=i
    ans =[]
    for i in range(len(nums)):
        ans.append(adic[nums[i]])
    return ans
print(smallerNumbersThanCurrent([3,4,3,3,3,1,2,6]))
"""

#1572. Matrix Diagonal Sum
"""def diagonalSum(mat):
    #sum=0
    #alen=len(mat)
    #amid=int((alen-1)/2)
    #for i in range(alen):
    #    sum+=mat[i][i]
    #    sum+=mat[i][alen-1-i]
    #return sum if alen%2==0 else sum-mat[amid][amid]
    alen = len(mat)
    res = 0
    for i in range(alen):
        res += mat[i][i] + mat[i][alen-1-i]
    amid = (alen-1)//2

    return res-mat[amid][amid] if alen%2 else res

print(diagonalSum([[1,1,1,1],
              [1,1,1,1],
              [1,1,1,1],
              [1,1,1,1]]))"""

#1389. Create Target Array in the Given Order

"""def createTargetArray(nums, index):
    ans=[]
    for i in range(len(nums)):
        ans.insert(index[i],nums[i])
    return ans
print(createTargetArray([0,1,2,3,4],[0,1,2,2,1]))"""

#1266. Minimum Time Visiting All Points
"""def minTimeToVisitAllPoints(points):
    ans=0
    for i in range(len(points)-1):
        ans+=max([abs(x-y) for x,y in zip(points[i],points[i+1])])
    return ans
print(minTimeToVisitAllPoints([[1,1],[3,4],[-1,0]]))
"""
"""
    res = 0
    x1, y1 = points.pop()
    while points:
        x2, y2 = points.pop()
        res += max(abs(y2 - y1), abs(x2-x1))
        x1, y1 = x2, y2
    return res
"""
#1512. Number of Good Pairs
"""import collections
def numIdenticalPairs(nums):
    adic=collections.Counter(nums)

    return int(sum([x*(x-1)/2 for x in adic.values()]))
print(numIdenticalPairs([1,1,1,1]))"""

#1534. Count Good Triplets
"""
def countGoodTriplets(arr, a,b,c):
    ans=0
    for i in range(len(arr)-2):
        for j in range(i+1,len(arr)-1):
            for k in range(j+1,len(arr)):
                if abs(arr[i]-arr[j])<=a and abs(arr[j]-arr[k])<=b and abs(arr[k]-arr[i])<=c:
                    ans+=1
    return ans
print(countGoodTriplets([3,0,1,1,9,7], 7, 2, 3))
"""

#1640. Check Array Formation Through Concatenation
"""def canFormArray(arr, pieces):
    amap={x[0]:x for x in pieces}
    res=[]
    
    for i in arr:
        res+=amap.get(i,[])
    return res==arr
    """

#1295. Find Numbers with Even Number of Digits
"""def findNumbers(nums):
    ans=list(map(lambda x: len(str(x))%2,nums))
    return len(nums)-sum(ans)
print(findNumbers( [555,901,482,1771]))"""

#1588. Sum of All Odd Length Subarrays
"""def sumOddLengthSubarrays(arr):
    def oddx(x):
        return (x//2+x%2)*2-1
    ans=0
    res=oddx(len(arr))
    i=(res+1)//2
    for x in range(i):
        gap=2*x+1
        tmp=0
        for i in range(len(arr)-gap+1):
            tmp+=sum(arr[i:i+gap])
        ans+=tmp
    return ans
print(sumOddLengthSubarrays([10,11,12]))"""
"""
 def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        n, sum_odd = len(arr), 0
        p_sum = [0] * ( n + 1)
        for i, a in enumerate(arr):
            p_sum[i + 1] = p_sum[i] + a
        for i, p in enumerate(p_sum):
            for j in range(i + 1, n + 1, 2):
                sum_odd += p_sum[j] - p_sum[i] 
        return sum_odd"""

#1486. XOR Operation in an Array
"""
def xorOperation(n, start):
    from functools import reduce
    res=[x*2+start for x in range(n)]

    return reduce(lambda x,y:x^y,res)
print(xorOperation(10,5))"""

#1313. Decompress Run-Length Encoded List
"""
def decompressRLElist(nums):
    ans=[]
    for i in range(0,len(nums),2):
        ans+=[nums[i+1] for x in range(nums[i])]
    return ans
print(decompressRLElist( [1,2,3,4]))
"""

#1252. Cells with Odd Values in a Matrix
"""
def oddCells(n, m, indices):
    ans=[[0 for x in range(m)] for y in range(n)]
    for op in indices:
        ans[op[0]]=list(map(lambda x:x+1,ans[op[0]]))
        for x in range(n):
            ans[x][op[1]]+=1
    tmp=0
    for i in range(n):
        tmp+=sum(map(lambda x:x%2==1,ans[i]))
    return tmp
print(oddCells(2,2,[[1,1],[0,0]]))
"""

#1436. Destination City
def destCity(paths):
    s=[]
    d=[]
    for x in paths:
        s.append(x[0])
        d.append(x[1])
    sset=set(s)
    dset=set(d)
    for i in dset:
        if i not in sset:
            return i
    
print(destCity([["B","C"],["D","B"],["C","A"]]))

        