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
"""
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
    
print(destCity([["B","C"],["D","B"],["C","A"]]))"""

#1395. Count Number of Teams
"""
def numTeams(rating):
    ans=0
    def lowhighthan(num,alist):
        return sum(map(lambda x:x<num, alist)), sum(map(lambda x:x>num, alist))
    for num in range(1,len(rating)-1):
        l=rating[:num]
        r=rating[num+1:]
        lres=lowhighthan(rating[num],l)
        rres=lowhighthan(rating[num],r)
        print(lres,rres)
        ans+=lres[0]*rres[1]+lres[1]*rres[0]
    return ans
print(numTeams([2,5,3,4,1]))
"""

#1450. Number of Students Doing Homework at a Given Time
"""
def busyStudent(startTime, endTime, queryTime):
    start=list(map(lambda x: x<=queryTime,startTime))
    end=list(map(lambda x: x>=queryTime,endTime))
    return sum(map(lambda x: x[0] and x[1], zip(start,end) ))
print(busyStudent( [9,8,7,6,5,4,3,2,1],[10,10,10,10,10,10,10,10,10],5))
"""

#1646. Get Maximum in Generated Array
"""
def getMaximumGenerated(n):
    if n==0:
        return 0
    elif n==1:
        return 0
    elif n==2:
        return 1
    else:
        res=[0,1]
        for i in range(2,n+1):
            if i%2==0:
                res.append(res[i//2])
            else:
                res.append(res[(i-1)//2]+res[(i-1)//2+1])
    return max(res)
print(getMaximumGenerated(2))"""

#1535. Find the Winner of an Array Game
"""def getWinner(arr, k):
    alen=len(arr)
    maxnum=max(arr)
    

    defender=arr[0]
    if defender==maxnum:
        return defender
    else:
        defending=True
        tmp=0
        challenger=1
        while defending:
            if defender>arr[challenger]:
                tmp+=1
                if tmp==k:
                    return defender
                challenger+=1
            else:
                defending=False
    for i in range(1,alen):
        if arr[i]>defender:
            tmp=1
            if tmp==k:
                    return arr[i]
            while 
"""  
    


#1652. Defuse the Bomb
"""
def decrypt(code, k):
    if len(code)==1:
        return [0]
    alen=len(code)
    if k==0:
        return [0 for x in code]
    elif k>0:
        ans=[]
        base=sum(code[1:k+1])
        ans.append(base)
        for i in range(1,len(code)):
            base-=code[(i)%alen]
            base+=code[(i+k)%alen]
            ans.append(base)
    else:
        ans=[]
        base=sum(code[alen+k:])
        print(base)
        ans.append(base)
        for i in range(1,len(code)):
            base-=code[(alen+k-1+i)%alen]
            print(base)
            base+=code[i-1]
            ans.append(base)
    return ans
print(decrypt([2,4,9,3],-2))
"""

#945. Minimum Increment to Make Array Unique
"""def minIncrementForUnique(A):
    alen=len(A)
    if alen==1:
        return 0
    res=sorted(A)
    mark=res[0]
    ans=0
    for i in range(1,alen):
        if res[i]<=mark:  
            mark+=1
            ans+=mark-res[i]
        mark=max(mark,res[i])
    return ans
print(minIncrementForUnique([3,1012,11,2,1,7]))"""

#1346. Check If N and Its Double Exist  
"""
def checkIfExist(arr):
    if 0 not in arr:
        return len(set(arr+list(map(lambda x:x*2, arr))))!=2*len(set(arr))
    else:
        res=[x for x in arr if x!=0]
        return len(set(res+list(map(lambda x:x*2, res))))!=2*len((res)) if len(arr)-len(res)==1 else True
print(checkIfExist([-2,0,100,-19,4,6,-8]))
"""

#1217. Minimum Cost to Move Chips to The Same Position
"""
def minCostToMoveChips(position):
    alen=len(position)
    if alen==1:
        return 0
    else:
        l=sum([1 for i in range(alen) if position[i]%2])
        r=sum([1 for i in range(alen) if not position[i]%2])
    return min(l,r)

print(minCostToMoveChips([1,2,3,4,5,6]))"""

#1508. Range Sum of Sorted Subarray Sums
"""def rangeSum(nums, n, left, right):
    res=[]
    for i in range(n):
        for j in range(i+1,n+1):
            res.append(sum(nums[i:j]))
    return sum(sorted(res)[left-1:right])
print(16716700000%(10**9+7))"""

#1465. Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts
def maxArea(h, w, horizontalCuts, verticalCuts):
    hori=sorted([0]+horizontalCuts+[h])
    vert=sorted([0]+verticalCuts+[w])
    print(hori,vert)
    h=max(map(lambda x:x[1]-x[0], zip(hori[:-1],hori[1:])))       
    v=max(map(lambda x:x[1]-x[0], zip(vert[:-1],vert[1:]))) 
    return (h*v)%(10**9+7)
print(maxArea(5,4,[3,1],[1]))