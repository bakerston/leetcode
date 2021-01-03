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
"""
def maxArea(h, w, horizontalCuts, verticalCuts):
    hori=sorted([0]+horizontalCuts+[h])
    vert=sorted([0]+verticalCuts+[w])
    print(hori,vert)
    h=max(map(lambda x:x[1]-x[0], zip(hori[:-1],hori[1:])))       
    v=max(map(lambda x:x[1]-x[0], zip(vert[:-1],vert[1:]))) 
    return (h*v)%(10**9+7)
print(maxArea(5,4,[3,1],[1]))"""
#1539. Kth Missing Positive Number
"""
def findKthPositive(arr, k):
    index=[x for x in range(1,arr[-1]+1) if x not in arr]
    if k>len(index):
        return arr[-1]+k-len(index)
    else:
        return index[k-1]
print(findKthPositive([1,2,3,4,5],5))
"""

#1464. Maximum Product of Two Elements in an Array
"""
def maxProduct(nums):
    res=[x-1 for x in nums]
    return sorted(res)[-1]*sorted(res)[-2]
print(maxProduct([3,4,5,2]))
"""

#1200. Minimum Absolute Difference
"""
def minimumAbsDifference(arr):
    brr=sorted(arr)
    diff=min(map(lambda x:x[0]-x[1],zip(brr[1:],brr[:-1])))
    return [[x[1],x[0]] for x in zip(brr[1:],brr[:-1]) if x[0]-x[1]==diff]
print(minimumAbsDifference([4,2,1,3]))"""

#509. Fibonacci Number
"""
def fib(N):
    dp=[0,1]
    def fb(n):
        if n<len(dp):
            return dp[n]
        else:
            temp_fb=fb(n-1)+fb(n-2)
            dp.append(temp_fb)
            return temp_fb
    return  fb(N)
print(fib(4))
"""
#1267. Count Servers that Communicate
"""
def countServers(grid):
    dis_dic={}
    con_dic={}
    ans=0
    for row in range(len(grid)):
        if sum(grid[row])==0:
            continue
        elif sum(grid[row])==1:
            for col in range(len(grid[row])):
                if grid[row][col]==1:
                    if col not in con_dic.keys():
                        if col not in dis_dic:
                            dis_dic[col]=1
                        else:
                            del dis_dic[col]
                            con_dic[col]=1
                            ans+=2
                    else:
                        ans+=1
                else:
                    continue
        else:
            ans+=sum(grid[row])
            for col in range(len(grid[row])):
                if grid[row][col]==1:
                    if col not in con_dic.keys():
                        if col not in dis_dic:
                            dis_dic[col]=1
                        else:
                            del dis_dic[col]
                            con_dic[col]=1
                    else:
                        continue
                else:
                    continue
    return ans
print(countServers([[1,0,0,1,0],[0,0,0,0,0],[0,0,0,1,0]]))
#print(countServers([[1,0,0,0,1,0,0],[1,0,0,0,0,0,0],[0,0,1,1,0,0,1],[0,0,0,0,0,1,0],[1,0,0,0,1,0,0],[0,0,0,0,1,0,0],]))

def countServers(grid):
    rowsum=[sum(x) for x in grid]
    colsum=[sum(row[x] for row in grid) for x in range(len(grid[0]))]
    r=[i for i in range(len(rowsum)) if rowsum[i]==1]
    c=[j for j in range(len(colsum)) if colsum[j]==1]
    ans=0
    for i in r:
        for j in c:
            if grid[i][j]==1:
                ans+=1
    res=sum([sum(x) for x in grid])

    return res-ans


print(countServers([[1,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1]]))
"""

#1550. Three Consecutive Odds
"""
def threeConsecutiveOdds(arr):
    res=[x%2 for x in arr]
    tmp=0
    print(res)
    for x in res:
        if x==1:
            tmp+=1
            if tmp==3:
                return True
        else:  
            tmp=0
    return False
print(threeConsecutiveOdds([1,2,34,3,4,5,4,7,23,12]))
"""

#1502. Can Make Arithmetic Progression From Sequence
"""
def canMakeArithmeticProgression(arr):
    return len(set(map(lambda x: x[0]-x[1],zip(sorted(arr)[1:],sorted(arr)[:-1]))))==1
print(canMakeArithmeticProgression([1,2,3,6,7,5,4]))
"""

#1560. Most Visited Sector in a Circular Track
"""def mostVisited(n,rounds): 
    start=rounds[0]
    end=rounds[-1]
    base=[x for x in range(1,n+1)]
    return base[start-1:end] if start<=end else base[:end]+base[start-1:]
print(mostVisited(7,[1,3,5,7]))
print(mostVisited(4,[3,1]))"""

#1051. Height Checker
"""
def heightChecker(heights):
    return sum(h1 != h2 for h1, h2 in zip(heights, sorted(heights)))"""

#922. Sort Array By Parity II
"""
def sortArrayByParityII(A):
    odd=[x for x in A if x%2==1]
    even=[x for x in A if x%2==0]
    ans=[]
    for x in range(len(odd)):
        ans.append(even[x])
        ans.append(odd[x])
    return ans
print(sortArrayByParityII([4,2,5,7]))    
"""
#1207. Unique Number of Occurrences
"""
def uniqueOccurrences(arr):   
    import collections
    adic=collections.Counter(arr)
    return len(list(adic.values())) == len(set(adic.values()))
print(uniqueOccurrences([1,1,1,2,2,3]))"""

#1403. Minimum Subsequence in Non-Increasing Order
"""
def minSubsequence(nums):
    asum=sum(nums)
    tar=int(sum(nums)/2)
    res=(sorted(nums))[::-1]
    i=0
    tmp=res[i]
    while tmp<=tar:
        i+=1
        tmp+=res[i]
    return res[:i+1]
print(minSubsequence([6]))"""

#1047. Remove All Adjacent Duplicates In String
"""
def removeDuplicates(S):
    stack=[]
    for i in S:
        if len(stack)>0:
            if i==stack[-1]:
                stack.pop()
            else:
                stack.append(i)
        else:
            stack.append(i)
    return "".join(stack)      
print(removeDuplicates("aabbaddae")) 
"""

#1385. Find the Distance Value Between Two Arrays
"""
def findTheDistanceValue( arr1, arr2, d):
    count = 0
    for x in arr1:
        for y in arr2:
            if abs(x-y) <= d:
                count += 1
                break
    return len(arr1) - count
"""


#1030. Matrix Cells in Distance Order
"""
def allCellsDistOrder(R, C, r0, c0):
    ans=[]
    for row in range(R):
        for col in range(C):
            ans.append([abs(row-r0)+abs(col-c0),row,col])
    res=sorted(ans,key=lambda x:x[0])
    return [x[1:] for x in res]
print(allCellsDistOrder(2,3,1,2))
"""

#1524. Number of Sub-arrays With Odd Sum
"""
def numOfSubarrays(arr):
    res=[x%2 for x in arr]
    tmp=[0]
    mark=0
    for x in range(len(res)):
        mark+=res[x]
        if mark%2==1:
            tmp.append(1)
            mark=1
        else:
            tmp.append(0)
            mark=0
    odd=sum(tmp)
    even=len(tmp)-odd
    ans=0
    for x in range(len(tmp)):
        if tmp[x]==0:
            ans+=odd
            even-=1
        else:
            ans+=even
            odd-=1
    return ans
print(numOfSubarrays([1,2,3,4,5,6,7]))"""
"""res=[x%2==1 for x in arr]
    ans= 0
    alen=len(res)
    for x in range(alen):
        tmp=res[x]
        if tmp:
            ans+=1
        for y in range(x+1,alen):
            new=res[y]
            if tmp:
                if new:
                    tmp=False
                else:
                    ans+=1
            else:
                if new:
                    tmp=True
                    ans+=1
    return ans

print(numOfSubarrays([1,2,3,4,5,6,7]))"""
"""
    alen=len(arr)
    ans=0
    for x in range(alen):
        tmp=arr[x]
        if tmp%2==1:
            ans+=1
        for y in range(x+1,alen):
            tmp+=arr[y]
            if tmp%2==1:
                ans+=1    
    return ans%(10**9+7)"""


#1031. Maximum Sum of Two Non-Overlapping Subarrays
"""
def maxSumTwoNoOverlap(A, L, M):
    alen=len(A)
    #leftL inclusive, rightM exclusive
    ans=0
    for i in range(L-1,alen-M):
        left=0
        right=0
        print(i-L+1)
        for j in range(i-L+2):
            left=max(left,sum(A[j:j+L]))
        for k in range(i+1,alen-M+1):
            right=max(right,sum(A[k:k+M]))
        ans=max(ans,left+right)
    #leftM inclusive, rightL exclusive
    for i in range(M-1,alen-L):
        left=0
        right=0
        for j in range(i-M+2):
            left=max(left,sum(A[j:j+M]))
        for k in range(i+1,alen-L+1):
            right=max(right,sum(A[k:k+L]))
        ans=max(ans,left+right)
    return ans
print(maxSumTwoNoOverlap(A = [0,6,5,2,2,5,1,9,4], L = 1, M = 2))
"""
#1304. Find N Unique Integers Sum up to Zero
"""def sumZero(n):
    base_odd=[1,0,-1]
    base_even=[1,-1]
    if n==1:
        return [0]
    elif n==2:
        return base_even
    elif n==3:
        return base_odd
    else:
        if n%2:
            k=int((n-3)/2)
            for i in range(1,k+1):
                base_odd.append(i+1)
                base_odd.append(-i-1)
            return base_odd
        else:
            k=int((n-2)/2)
            for i in range(1,k+1):
                base_even.append(i+1)
                base_even.append(-i-1)
            return base_even
print(sumZero(10))"""

#1144. Decrease Elements To Make Array Zigzag
"""
def movesToMakeZigzag(nums):
    if len(nums)==1:
        return 0
    odd,even=0,0
    for i in range(0,len(nums),2):
        if i==0:
            k=nums[i]-nums[i+1]
            if k>=0:
                odd+=k+1
        elif i+1==len(nums):
            k=nums[i]-nums[i-1]
            if k>=0:
                odd+=k+1
        else:
            k=max(nums[i]-nums[i-1],nums[i]-nums[i+1])
            if k>=0:
                odd+=k+1
    for i in range(1,len(nums),2):
        if i+1==len(nums):
            k=nums[i]-nums[i-1]
            if k>=0:
                even+=k+1
        else:
            k=max(nums[i]-nums[i-1],nums[i]-nums[i+1])
            if k>=0:
                even+=k+1
    return min(odd,even)
print(movesToMakeZigzag([2,7,10,9,8,9]))  
"""
#1014. Best Sightseeing Pair
"""
def maxScoreSightseeingPair(A):
    i,j=0,1
    alen=len(A)
    ans=0
    while j<alen:
        ans=max(ans,A[i]+A[j]-(j-i))
        if A[i]-A[j]<=j-i:
            i=j
        j+=1
    return ans
print(maxScoreSightseeingPair([8,1,5,2,6]))
"""


#1208. Get Equal Substrings Within Budget

"""
def equalSubstring(s, t, maxCost):
    res=[abs(ord(x[0])-ord(x[1])) for x in zip(list(s),list(t))]
    print(res)
    i=0
    j=0
    ans=0
    cur=0
    while j<len(res):
        if cur<=maxCost:
            ans=max(ans,j-i)
            cur+=res[j]
            j+=1
        else:
            while cur>maxCost:
                cur-=res[i]
                i+=1
    if cur<=maxCost:
        ans=max(ans,j-i)
    return ans
print(equalSubstring(s = "abcd", t = "bcdf", maxCost = 3))
print(equalSubstring(s = "abcd", t = "cdef", maxCost = 3))
print(equalSubstring("abcd", t = "acde", maxCost = 0))
print(equalSubstring("zdfy","sgby",14))
"""

#1375. Bulb Switcher III
"""
def numTimesAllBlue(light):
    alen=len(light)
    ans=0
    blue=0
    for i in range(alen):
        blue=max(blue,light[i])
        if blue==i+1:
            ans+=1
    return ans
print(numTimesAllBlue([2,1,4,3,6,5]))
"""

#1423. Maximum Points You Can Obtain from Cards
"""
def maxScore(cardPoints, k):
    base=sum(cardPoints[-k:])
    res=0
    ans=0
    for i in range(k):
        res+=cardPoints[i]-cardPoints[-k+i]
        ans=max(ans,res)
    return ans+base
print(maxScore(cardPoints = [1,79,80,1,1,1,200,1], k = 3))
"""

#1122. Relative Sort Array
"""
def relativeSortArray(arr1, arr2):
    import collections
    dic1=collections.Counter(arr1)
    dic2=collections.Counter(arr2)
    ans=[]
    for x in arr2:
        ans+=dic1[x]*[x]
    miss=[x for x in dic1.keys() if x not in dic2.keys()]
    for x in miss:
        ans+=dic1[x]*[x]
    return ans
print(relativeSortArray( arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]))
"""

#1337. The K Weakest Rows in a Matrix
"""
def kWeakestRows(mat, k):
    res=[sum(x) for x in mat]
    adic=[]
    for x in range(len(res)):
        adic.append([res[x],x]) 
    b=sorted(adic,key=lambda x: [x[0],x[1]])
    return [x[1] for x in b][:k]
print(kWeakestRows(mat = 
[[1,0,0,0],
 [1,1,1,1],
 [1,0,0,0],
 [1,0,0,0]], 
k = 2))
"""
#1170. Compare Strings by Frequency of the Smallest Character
"""
def numSmallerByFrequency(queries, words):
    import collections
    def smallfreq(str):
        tmp=collections.Counter(list(str))
        return tmp[sorted(list(tmp.keys()))[0]]
    tmp=[smallfreq(x) for x in queries]
    wor=[smallfreq(x) for x in words]
    ans=[sum([1 for x in wor if x>y]) for y in tmp]
    return ans
print(numSmallerByFrequency(queries = ["cbd"], words = ["zaaaz"]))
"""

#1481. Least Number of Unique Integers after K Removals
"""
def findLeastNumOfUniqueInts(arr,k):
    if k==len(arr):
        return 0
    import collections
    res=collections.Counter(arr)
    tmp=sorted(list(res.values()))
    alen=len(tmp)
    i=0
    while k>=0:
        k-=tmp[i]
        i+=1
    return alen-i+1
print(findLeastNumOfUniqueInts(arr = [4,3,1,1,3,3,2], k = 3))
print(findLeastNumOfUniqueInts(arr = [5,5,4,4,4,6,6,6,6,7,7,7,7,7], k = 10))
print(findLeastNumOfUniqueInts([1,1,2,2,2],3))"""

#1619. Mean of Array After Removing Some Elements
"""
def trimMean(arr):
    tmp=sorted(arr)
    sec=int(len(tmp)/20)
    print(sec)
    return (sum(arr)-sum(tmp[:sec])-sum(tmp[-sec:]))/(len(tmp)-2*sec)
print(trimMean(arr = [4,8,4,10,0,7,1,3,7,8,8,3,4,1,6,2,1,1,8,0,9,8,0,3,9,10,3,10,1,10,7,3,2,1,4,9,10,7,6,4,0,8,5,1,2,1,6,2,5,0,7,10,9,10,3,7,10,5,8,5,7,6,7,6,10,9,5,10,5,5,7,2,10,7,7,8,2,0,1,1]))
"""
#1437. Check If All 1's Are at Least Length K Places Away
"""
def kLengthApart(nums,k):
    if k==0:
        return True
    else:
        res=[x for x in range(len(nums)) if nums[x]==1]
        if len(res)<=1:
            return True
        else:
            dif=list(map(lambda x: x[1]-x[0], zip(res[:-1],res[1:])))
            return min(dif)>=k+1
print(kLengthApart(nums = [1,1,1,1,1], k = 0))"""

#1414. Find the Minimum Number of Fibonacci Numbers Whose Sum Is K
"""
def findMinFibonacciNumbers(k):
    def maxfib(n):
        if n<=2:
            return n
        else:
            a,b,ans,k=1,1,1,2
            while ans<=n:
                a,b=b,a+b
                ans=b
                k+=1
            return a
    ans=0
    while k>0:
        k-=maxfib(k)
        ans+=1
    return ans

print(findMinFibonacciNumbers(143))
"""

#1013. Partition Array Into Three Parts With Equal Sum
#trash question, fk the trash who made this shit problem.
"""
def canThreePartsEqualSum(A):
    t=sum(A)
    if t%3!=0:
        return False 
    else:
        l=False
        r=False
        ans=0
        for x in A:
            ans+=x
            if ans==int(t/3):
                l=True
            elif ans==int(2*t/3):
                r=True
        return l and r
print(canThreePartsEqualSum( [0,2,1,-6,6,-7,9,1,2,0,1]))
"""

#1544. Make The String Great
"""
def makeGood(s):
    base=list(s)
    stack=[]
    for i in base:
        if not stack:
            stack.append(i)
        else:
            if abs(ord(i)-ord(stack[-1]))==32:
                stack.pop(-1)
            else:
                stack.append(i)
    return "".join(stack)
print(makeGood("as"))
"""

#1287. Element Appearing More Than 25% In Sorted Array
"""
def findSpecialInteger(arr):
    import collections
    res=collections.Counter(arr)
    a=max(res.items(),key=lambda x: x[1])
    return a[0]
print(findSpecialInteger( arr = [1,2,2,6,6,6,6,7,10]))
"""

#1482. Minimum Number of Days to Make m Bouquets
"""
def minDays(bloomDay, m, k):
    l=min(bloomDay)
    r=max(bloomDay)+1
    def isgood(mid):
        count=0
        cur=0
        for day in bloomDay:
            if day>mid:
                cur=0
            else:
                cur+=1
                if cur==k:
                    cur=0
                    count+=1
        return count>=m
    while l<r:
        mid=l+(r-l)//2
        if isgood(mid):
            r=mid
        else:
            l=mid+1
    return l if l <=max(bloomDay) else -1


    from itertools import groupby
    days=sorted(list(set(bloomDay)))
    t=[]
    def makebou(lis,m,k):
        a=[len(x) for x in lis]
        return sum([x//k for x in a])>=m
    succ=len(bloomDay)>=m*k
    if not succ:
        return -1
    else:
        while succ:
            t.append(days.pop())
            cur=[list(g) for k, g in groupby(bloomDay, lambda x: x in t) if not k]
            succ=makebou(cur,m,k)
        return t[-1]
"""

#1380. Lucky Numbers in a Matrix
"""
def luckyNumbers (matrix):
    ans=[]
    for x in range(len(matrix)): 
        row=[[k,g] for k,g in enumerate(matrix[x])]
        tmp=sorted(row,key=lambda x:x[1])[0]
        base=[matrix[x][tmp[0]] for x in range(len(matrix))]
        if tmp[1]==max(base):
            ans.append(tmp[1])       
    return ans
print(luckyNumbers(matrix = [[7,8],[1,2]]))
"""
         
#1370. Increasing Decreasing String
"""
def sortString(s):
    import collections
    base=collections.Counter(list(s))
    res=[list(x) for x in base.items()]
    b=sorted(res,key=lambda x: ord(x[0]))
    ans=""
    n=len(list(s))
    print(b)
    while n>0:
        for x in range(len(b)):
            if b[x][1]>0:
                ans+=(b[x][0])
                b[x][1]-=1
            
        for x in range(len(b)-1,-1,-1):
            if b[x][1]>0:
                ans+=(b[x][0])
                b[x][1]-=1         
        n-=1
    return ans

print(sortString("aaaabbbbcccc"))
"""

#1408. String Matching in an Array
"""
def stringMatching(words):
    if len(words)==1:
        return []
    else:
        ans=[]
        base=sorted(words,key=lambda x:len(x))
        for x in range(len(base)-1):
            for y in range(x+1,len(base)):
                if len(base[y])>len(base[x]):
                    if base[y].find(base[x])!=-1:
                        ans.append(base[x])
                        break
    return ans
print(stringMatching(  ["leetcoder","leetcode","od","hamlet","am"]))
"""

#1331. Rank Transform of an Array
"""
def arrayRankTransform(arr):
    import collections
    base=collections.Counter(arr)
    n=len(list(base.keys()))
    tmp=[x for x in range(1,n+1)]
    new=list(zip(sorted(list(base.keys())),tmp))
    adic={x:y for x,y in new}
    for x in range(len(arr)):
       arr[x]=adic[arr[x]]
    return arr
print(arrayRankTransform(arr = [37,12,28,9,100,56,80,5,12]))
"""

#1526. Minimum Number of Increments on Subarrays to Form a Target Array
"""
def minNumberOperations(target):    
    if len(target)==1:
        return sum(target)
    else:
        base=[x for x in list(map(lambda x:x[0]-x[1], zip(target[1:],target[:-1]))) if x<0]
        return -sum(base)+target[-1]
print(minNumberOperations(target = [3,1,1,2]))

    from itertools import groupby
    n=max(target)
    l=len(target)
    ans=0
    for x in range(n):
        tmp=[]
        for i in range(l):
            if target[i]>0:
                tmp.append(1)
                target[i]-=1
            else:
                tmp.append(0)
        print(target)
        print(tmp)
        ans+=sum([x for x,k in groupby(tmp)])
    return ans
"""

#1636. Sort Array by Increasing Frequency
"""
def frequencySort(nums):
    import collections
    import functools
    base=collections.Counter(nums)
    c=sorted(list(base.items()),key=lambda x: [x[1],-x[0]])
    return list(functools.reduce(lambda x,y:x+y, map(lambda x: [x[0]]*x[1],c)))
    
print(frequencySort([1,1,2,2,2,3]))
"""

#1629. Slowest Key
"""
def slowestKey(releaseTimes, keysPressed):
    if len(releaseTimes)==1:
        return keysPressed
    else:
        dic={keysPressed[0]:releaseTimes[0]}
        l=len(releaseTimes)
        for x in range(1,l):
            if keysPressed[x] not in dic.keys():
                dic[keysPressed[x]]=releaseTimes[x]-releaseTimes[x-1]
            else: 
                dic[keysPressed[x]]=max(dic[keysPressed[x]], releaseTimes[x]-releaseTimes[x-1])
    
    return sorted(dic.items(),key=lambda x: [x[1],x[0]])[-1][0]
print(slowestKey( releaseTimes = [12,23,36,46,62], keysPressed = "spuda"))
"""

#57. Insert Interval
"""
def insert(intervals, newInterval):
    left,right=[],[]
    l,r=newInterval
    for x in intervals:
        if x[1]<l:
            left+=[x]
        elif x[0]>r:
            right+=[x]
        else:
            l=min(l,x[0])
            r=max(r,x[1])
    return left+[[l,r]]+right
print(insert(intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]))
"""
#1630. Arithmetic Subarrays
"""
def checkArithmeticSubarrays(nums, l, r):
    def isarith(a):
        if len(a)<2:
            return False
        else:
            return len(set(map(lambda x: x[1]-x[0],zip(sorted(a)[1:],sorted(a)[:-1]))))==1
    return [isarith(nums[l[x]:r[x]+1]) for x in range(len(l))]   
print(checkArithmeticSubarrays(nums = [-12,-9,-3,-12,-6,15,20,-25,-20,-15,-10], l = [0,1,6,4,8,7], r = [4,4,9,7,9,10])) 
"""
#1509. Minimum Difference Between Largest and Smallest Value in Three Moves
"""def minDifference(nums):
    import collections
    base=collections.Counter(nums)
    c=sorted(list(base.items()),key=lambda x:x[0])

    return c
print(minDifference(nums = [6,6,0,1,1,4,6]))
"""

#1156. Swap For Longest Repeated Character Substring
"""
def maxRepOpt1(text):
    import itertools
    import collections
    base=[[k,len(list(g))] for k,g in itertools.groupby(text)]
    dic=collections.Counter(text)
    ans=max([x[1] for x in base])
    if len(base)<=2:
        return ans
    else:
        for l,m,r in zip(base[:-2],base[1:-1],base[2:]):
            if l[0]==r[0]:
                if m[1]==1:
                    if dic[l[0]]>l[1]+r[1]:
                        ans=max(ans,l[1]+r[1]+1)
                    else:
                        ans=max(ans,l[1]+r[1])
                else:
                    ans=max(ans,max(l[1],r[1])+1)
    c=max([x[1]for x in base])
    tmp=[x for x in dic.keys() if [x,c] in base]
    for x in tmp:
        if dic[x]>c:
            ans=max(ans,c+1)
    return ans
print(maxRepOpt1(  text = "abcdaaa"))
"""
#1004. Max Consecutive Ones III
"""
def longestOnes(A, K):
    import itertools
    base=[[k,len(list(g))] for k,g in itertools.groupby(A)]
    ans=0
    countb=[x for x in base if x[0]==1]
    if len(countb)<1:
        return K
    elif len(countb)<2:
        return countb[0]+min(K,sum(A)-countb)
    else:
        if base[0][0]==0:
            i,j=1,3
        else:
            i,j=0,2
        print(i,j)
        ans=base[i][1]
        tmp=base[i][1]
        used_0=0
        while j<len(base):
            if base[j-1][1]<K:
                if K-used_0>=base[j-1][1]:
                    tmp+=base[j][1]+base[j-1][1]
                    ans=max(ans,tmp)
                    used_0+=base[j-1][1]
                    j+=2
                else:
                    tmp-=base[i][1]-base[i+1][1]
                    used_0-=base[i+1][1]
                    i+=2    
            else:
                i=j
                j+=2
                used_0=0
                tmp=base[i][1]
                ans=max(ans,tmp)
    return ans

print(longestOnes( A = [1,1,1,0,0,0,1,1,1,1,0], K = 2))
"""

#1497. Check If Array Pairs Are Divisible by k
"""
def canArrange(arr, k):
    import collections
    c=[x%k for x in arr if x%k!=0]
    if len(c)%2!=0:
        return False
    dic=collections.defaultdict(int)
    for x in c:
        if dic[k-x]!=0:
            dic[k-x]-=1
        else:
            dic[x]+=1
    return max(dic.values())==0 if len(dic.values())>0 else True
print(canArrange(arr = [], k = 2))
"""
#1352. Product of the Last K Numbers
"""
class ProductOfNumbers:

    def __init__(self):
        self.l=[1]
    def add(self, num: int) -> None:
        if num==0:
            self.l=[1]
        else:
            self.l.append(num*self.l[-1])          
    def getProduct(self, k: int) -> int:
        if k>=len(self.l):
            return 0
        else:
            return self.l[-1]//self.l[-k-1]
"""
#1424. Diagonal Traverse II
"""
def findDiagonalOrder(nums):
    import collections
    d=collections.defaultdict(list)
    for i in range(len(nums)):
        for j in range(len(nums[0])):
            d[i+j].append(nums[i][j])
    return [v for k in d.keys() for v in reversed(d[k])]
"""

#1545. Find Kth Bit in Nth Binary String
"""
def unhappyFriends(n, preferences, pairs):
    d={}
    alen=len(preferences)
    blen=len(preferences[0])
    for i in range(alen):
        d[i]={}
        for j in range(blen):
            d[i][preferences[i][j]]=j
    ans=[]
    plen=len(pairs)
    for x in range(plen-1):
        for y in range(x+1,plen):
            if (d[pairs[x][0]][pairs[y][0]]<d[pairs[x][0]][pairs[x][1]] and d[pairs[y][0]][pairs[x][0]]<d[pairs[y][0]][pairs[y][1]]) or (d[pairs[x][0]][pairs[y][1]]<d[pairs[x][0]][pairs[x][1]] and d[pairs[y][1]][pairs[x][0]]<d[pairs[y][1]][pairs[y][0]]):
                ans.append(pairs[x][0])
    for x in range(plen-1):
        for y in range(x+1,plen):
            if (d[pairs[x][1]][pairs[y][0]]<d[pairs[x][1]][pairs[x][0]] and d[pairs[y][0]][pairs[x][1]]<d[pairs[y][0]][pairs[y][1]]) or (d[pairs[x][1]][pairs[y][1]]<d[pairs[x][1]][pairs[x][0]] and d[pairs[y][1]][pairs[x][1]]<d[pairs[y][1]][pairs[y][0]]):
                ans.append(pairs[x][1])
    for y in range(plen-1):
        if (d[pairs[plen-1][0]][pairs[y][0]]<d[pairs[plen-1][0]][pairs[plen-1][1]] and d[pairs[y][0]][pairs[plen-1][0]]<d[pairs[y][0]][pairs[y][1]]) or (d[pairs[plen-1][0]][pairs[y][1]]<d[pairs[plen-1][0]][pairs[plen-1][1]] and d[pairs[y][1]][pairs[plen-1][0]]<d[pairs[y][1]][pairs[y][0]]):
            ans.append(pairs[y][0])
            break
    for y in range(plen-1):
        if (d[pairs[plen-1][1]][pairs[y][0]]<d[pairs[plen-1][1]][pairs[-1][0]] and d[pairs[y][0]][pairs[plen-1][1]]<d[pairs[y][0]][pairs[y][1]]) or (d[pairs[plen-1][1]][pairs[y][1]]<d[pairs[plen-1][1]][pairs[-1][0]] and d[pairs[y][1]][pairs[plen-1][1]]<d[pairs[y][1]][pairs[y][0]]):
            ans.append(pairs[y][1])
            break
    print(ans)
    return len(set(ans))
print(unhappyFriends(6,
[[1,4,3,2,5],[0,5,4,3,2],[3,0,1,5,4],[2,1,4,0,5],[2,1,0,3,5],[3,4,2,0,1]],
[[3,1],[2,0],[5,4]]))
"""
#1124. Longest Well-Performing Interval


"""
def longestWPI(hours):
    import collections
    cur=0
    n=len(hours)
    s=collections.defaultdict(int)
    ans=0
    for i in range(n):
        if hours[i]>8:
            cur+=1
        else:
            cur-=1 
        if cur>0:
            ans=i+1
        s[cur]=i
        if cur-1 in s:
            print(cur-1)
            ans=max(ans,i-s[cur-1])
    return ans
print(longestWPI(hours = [9,9,6,0,6,6,9]))
"""

#1218. Longest Arithmetic Subsequence of Given Difference
"""
def longestSubsequence(arr, difference):
    import collections
    d=collections.defaultdict(int)
    for x in arr:
        if x-difference not in d:
            d[x]=1
        else:
            d[x]=d.pop(x-difference)
            d[x]+=1
    return max(list(d.values()))
print(longestSubsequence(arr = [1,5,7,8,5,3,4,2,1], difference = -2))
"""

#886. Possible Bipartition
"""
def possibleBipartition(N, dislikes):
    s1=set()
    s2=set()
"""

#624. Maximum Distance in Arrays
"""
def maxDistance(arrays):
    import collections
    s=[x[0] for x in arrays]
    b=[x[-1] for x in arrays]
    s1,s2=sorted(s)[0],sorted(s)[1]
    b1,b2=sorted(b)[-1],sorted(b)[-2]
    sc=collections.Counter(s)
    bc=collections.Counter(b)
    if sc[s1]==1 and bc[b1]==1 and (s1,b1) in zip(s,b):
        return max(b2-s1,b1-s2)
    else:
        return b1-s1
print(maxDistance(arrays = [[1],[1]]))
"""

#625. Minimum Factorization
"""
def smallestFactorization(a):
    ans=""
    while a>9:
        for x in range(9,1,-1):
            while a%x==0:
                ans+=str(x)
                a/=x
    return int(ans[::-1])

print(smallestFactorization(486))

"""












"""
def longestWPI(hours):
    if len(hours)==1:
        return 1 if hours[0]<=8 else 0
    else:
        alen=len(hours)
        base=[-1 if x<=8 else 1 for x in hours]
        tmp=0
        ans=[0]
        for x in base:
            tmp+=x
            ans.append(tmp)
        res=0
        for x in range(alen):
            for y in range(x+1,alen+1):
                if ans[y]-ans[x]>0:
                    res=max(res,y-x)
        return res

print(longestWPI(hours = [6,6,6,6,6,6]))
"""


#1598. Crawler Log Folder
"""
def minOperations(logs):
    cur=0
    for x in logs:
        if x=='./':
            continue
        elif x=='../':
            cur=max(cur-1,0)
        else:
            cur+=1
            
    return cur
print(minOperations(logs = ["d1/","d2/","./","d3/","../","d31/"]))
"""
#969. Pancake Sorting
"""
def pancakeSort(arr):
    ans=[]
    for i in range(len(arr),1,-1):
        x=arr.index(i)
        ans.extend([x+1,i])
        arr=arr[i-1:x:-1]+arr[:x+1]+arr[i:]
    return ans
print(pancakeSort([3,2,4,1]))
"""


#1109. Corporate Flight Bookings
"""
def corpFlightBookings(bookings, n):
    ans=[0]*(n+1)
    for i,j,k in bookings:
        ans[i-1]+=k
        ans[j]-=k
    res=0
    print(ans)
    for i in range(1,len(ans)):
        ans[i]+=ans[i-1]
    return ans[:-1]
print(corpFlightBookings( bookings = [[1,2,10],[2,3,20],[2,5,25]], n = 5))
"""

#3Sums Closest
"""
def threeSumClosest(nums, target):
    diff=float('inf')
    nums.sort()
    for i in range(len(nums)-2):
        l=i+1
        r=len(nums)-1
        while l<r:
            print(l,r)
            s=nums[i]+nums[l]+nums[r]
            if abs(s-target)<diff:
                diff=abs(s-target)
                ans=s
            if s<target:
                l+=1
            elif s>target:
                r-=1
            else:
                return ans
    return ans
print(threeSumClosest( nums = [-1,2,1,-4], target = 1))
"""
#1402. Reducing Dishes
"""
def maxSatisfaction(satisfaction):
    satisfaction.sort()
    cur_sum=0
    ans=0
    for x in satisfaction[::-1]:
        cur_sum+=x
        if cur_sum>=0:
            ans+=cur_sum
        else:
            break
    return ans
print(maxSatisfaction(satisfaction = [-2,5,-1,0,3,-3]))
"""

#1282.Group the People Given the Group Size They Belong To
"""
def groupThePeople(groupSizes):
    import collections
    d=collections.defaultdict(list)
    n=len(groupSizes)
    for i in range(n):
        d[groupSizes[i]].append(i)
    ans=[]
    print(d)
    for x in d.keys():
        tmp=[]
        for i in range(len(d[x])):
            tmp.append(d[x][i])
            if (i+1)%x==0:
                ans.append(tmp)
                tmp=[]
    return ans
print(groupThePeople(groupSizes = [3,3,3,3,3,1,3]))
"""

#763. Partition Labels
"""
def partitionLabels(S):
    import collections
    d=collections.defaultdict(list)
    n=len(S)
    for i in range(n):
        if len(d[S[i]])==2:
            d[S[i]][-1]=i
        else:
            d[S[i]].append(i)
    base=sorted(d.values(),key=lambda x: x[0])
    if len(base)==1:
        return n
    else:
        ans=[0]
        cur_l,cur_r=base[0][0],base[0][-1]
        print(base)
        for x in base[1:]:
            if x[0]<cur_r:
                cur_r=max(x[-1],cur_r)
            else:
                ans.append(x[0]-sum(ans))
                cur_l,cur_r=x[0],x[-1]
        ans.append(n-sum(ans))
        return ans[1:]
print(partitionLabels(S = "dccccbaabe"))
print(partitionLabels(S = "ababcbacadefegdehijhklij"))
"""

#1338. Reduce Array Size to The Half
"""
def minSetSize(arr):
    import collections
    import math
    base=collections.Counter(arr)
    n=len(arr)
    half=math.ceil(n/2)
    x=sorted(list(base.values()))[::-1]
    cur=0
    ans=0
    for i in x:
        cur+=i
        ans+=1
        if cur>=half:
            return ans

print(minSetSize(arr = [1,2,3,4,5,6,7,8,9,10]))
"""

#1433. Check If a String Can Break Another String
"""
def checkIfCanBreak(s1,s2):
    import collections
    b1=collections.Counter(s1)
    b2=collections.Counter(s2)
    base1=sorted(b1.items())
    base2=sorted(b2.items())
    str1,str2="",""
    for x in base1:
        str1+=x[0]*x[1]
    for x in base2:
        str2+=x[0]*x[1]
    ans1=[ord(str1[x])>=ord(str2[x]) for x in range(len(str1))]
    ans2=[ord(str2[x])>=ord(str1[x]) for x in range(len(str1))]
    return len(set(ans1))==1 or len(set(ans2))==1
print(checkIfCanBreak("szy","cid"))
"""

#973. K Closest Points to Origin
"""
def kClosest(points, K):
    ans=[x+[x[0]**2+x[1]**2] for x in points]
    ans.sort(key=lambda x: x[2])
    return [x[:-1] for x in ans[:K]]
print(kClosest(points = [[3,3],[5,-1],[-2,4]], K = 2))
"""

#1249. Minimum Remove to Make Valid Parentheses
"""
def minRemoveToMakeValid(s):
    ans=""
    cur=0
    for x in s:
        if x=="(":
            cur+=1
            ans+=x
        elif x==")":
            if cur>0:
                ans+=x
                cur-=1
            else:
                continue
        else:
            ans+=x
    cur=0
    tmp=""
    for x in ans[::-1]:
        if x==")":
            cur+=1
            tmp+=x
        elif x=="(":
            if cur>0:
                tmp+=x
                cur-=1
            else:
                continue
        else:
            tmp+=x
    return tmp[::-1]
print(minRemoveToMakeValid("))(("))
"""

#941. Valid Mountain Array
"""
def validMountainArray(arr):
    if len(arr)<3:
        return False
    else:
        d=max(arr)
        n=arr.index(d)
        if n==0 or n==len(arr)-1:
            return False
        else:
            for i in range(n):
                if arr[i]>=arr[i+1]:
                    return False
            for i in range(n,len(arr)-1):
                if arr[i]<=arr[i+1]:
                    return False
            return True
print(validMountainArray(arr = [0,3,2,1]))
"""

#950. Reveal Cards In Increasing Order
"""
def deckRevealedIncreasing(deck):
    deck.sort(reverse=True)
    s=[]
    for x in deck:
        s=s[-1:]+s[:-1]
        s=[x]+s
    return s
print(deckRevealedIncreasing([17,13,11,2,3,5,7]))
"""

#963. Minimum Area Rectangle II
"""
def minAreaFreeRect(points):
    s=float('inf')
    st={(x,y) for x,y in points}
    n=len(points)
    for i in range(n):
        x1,y1=points[i]
        for j in range(i+1,n):
            x2,y2=points[j]
            for k in range(j+1,n):
                x3,y3=points[k]
                if not (x3-x1)*(x2-x1)+(y3-y1)*(y2-y1) and (x2+x3-x1,y3+y2-y1) in st:
                    s=min(s,((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 * ((x3 - x1) ** 2 + (y3 - y1) ** 2) ** 0.5)
    return s if s<float('inf') else 0
print(minAreaFreeRect([[0,1],[2,1],[1,1],[1,0],[2,0]]))
"""

#997. Find the Town Judge
"""def findJudge(N, trust):
    
    import collections
    d=collections.defaultdict(list)
    for x in trust:
        d[x[0]].append(x[1])
    p=[]
    a=[]
    for x in d.keys():
        p.extend([x])
        a.extend(d[x])
    allpp=collections.Counter(a)
    if len(set(p))==N-1:
        if len(set(p+a))==N:
            key=list(set(p+a)-set(p))[0]
            return key if allpp[key]==N-1 else -1
        return -1
    return -1

print(findJudge( N = 3, trust = [[1,3],[2,3],[3,1]]))
"""



#1441. Build an Array With Stack Operations
"""
def buildArray(target, n):
    ans=[]
    cur=0
    for x in target:
        if x-cur>1:
            ans.extend(["Push","Pop"]*(x-cur-1))
        ans.extend(["Push"])
        cur=x
    return ans
print(buildArray(target = [2,3,4], n = 4))
"""

#1422. Maximum Score After Splitting a String
"""
def maxScore(s):
    on=sum(map(int,list(s)))
    ze=len(s)-on
    ans=0
    l=0
    r=on
    for x in range(len(s)-1):
        if s[x]=="1":
            r-=1
        else:
            l+=1
        ans=max(ans,l+r)
    return ans
print(maxScore(s = "1111"))
"""
#1446. Consecutive Characters
"""
def maxPower(s):
    import itertools
    return max([len(list(g)) for k,g in itertools.groupby(s)])
print(maxPower("leetcode"))
"""

#1128. Number of Equivalent Domino Pairs
"""
def numEquivDominoPairs(dominoes):
    import collections
    d=collections.defaultdict(int)
    for x in dominoes:
        if x[0]!=x[1]:
            d[str(min(x))+str(max(x))]+=1
        else:
            d[str(x[0])*2]+=1
    return sum(map(lambda x:x*(x-1)//2,list(d.values())))
print(numEquivDominoPairs(dominoes = [[1,2],[2,1],[3,4],[5,6]]))
"""

#1503. Last Moment Before All Ants Fall Out of a Plank
"""
def getLastMoment(n, left, right):
    if not left:
        return n-min(right)
    if not right:
        return max(left)
    return max(max(left),n-min(right))
print(getLastMoment(n = 9, left = [5], right = [4]))
"""

#825. Friends Of Appropriate Ages
#unsolved

#1360. Number of Days Between Two Dates
"""
def daysBetweenDates(date1,date2):
    d1={"01":31,"02":28,"03":31,"04":30,"05":31,"06":30,"07":31,"08":31,"09":30,"10":31,"11":30,"12":31}
    d2={"01":31,"02":29,"03":31,"04":30,"05":31,"06":30,"07":31,"08":31,"09":30,"10":31,"11":30,"12":31}
    a=date1.split("-")
    b=date2.split("-")
    a_i=int("".join(a))
    b_i=int("".join(b))
    new=max(a_i,b_i)
    old=min(a_i,b_i)
    if new==old:
        return 0
    else:


    return 
print(daysBetweenDates(date1 = "2019-06-29", date2 = "2019-06-30"))
"""

#767. Reorganize String
"""
def reorganizeString(S):
    import collections
    import math
    n=len(S)
    base=collections.Counter(S).most_common()
    if not collections.Counter(S).most_common()[0][1]<=math.ceil(len(S)/2):
        return ""
    else:
        tmp=""
        for x in base:
            tmp+=x[0]*x[1]
        if n%2:
            ans=""
            for x in range(n//2):
                ans+=tmp[x]+tmp[x+n//2+1]
            ans+=tmp[n//2]
            return ans
        else:
            ans=""
            for x in range(n//2):
                ans+=tmp[x]+tmp[x+n//2]
            return ans
print(reorganizeString("vvvlo"))
"""

#769. Max Chunks To Make Sorted
"""
def maxChunksToSorted(arr):
    l=float('inf')
    h=float('-inf')
    cur_bound=0
    cur_len=0
    ans=0
    for x in range(len(arr)):

        l=min(l,arr[x])
        h=max(h,arr[x])
        cur_len+=1
        print("l=",l,"h=",h,"cur_len=",cur_len)
        if l==cur_bound and cur_len==h-l+1:
            ans+=1
            l=float('inf')
            h=float('-inf')
            cur_bound+=cur_len
            cur_len=0
            print("---if","l=",l,"h=",h,"cur_len=",cur_len)
    return ans
print(maxChunksToSorted(arr = [1,0,2,3,4]))
"""

#1460. Make Two Arrays Equal by Reversing Sub-arrays
"""
def canBeEqual(target, arr):
    import collections
    return collections.Counter(target)==collections.Counter(arr)
print(canBeEqual(target = [1,2,3,3,4], arr = [2,3,4,1,3]))
"""

#528. Random Pick with Weight
"""
import random
import bisect
class Solution:
    def __init__(self, w):
        self.w=w
        self.weight=[x/sum(w) for x in w]
        self.ans=[]
        tmp=0
        for x in self.weight:
            tmp+=x
            self.ans.append(tmp)

    def pickIndex(self) -> int:
        a=random.randint(0,1)
        mark=bisect.bisect_left(self.ans,a)
        return self.weight[mark]

a=Solution([1,3,5])
print(a.pickIndex())
print(a.pickIndex())
print(a.pickIndex())
print(a.pickIndex())
print(a.pickIndex())
print(a.pickIndex())
print(a.pickIndex())
print(a.pickIndex())
print(a.pickIndex())
print(a.pickIndex())
print(a.pickIndex())
"""

#842. Split Array into Fibonacci Sequence
"""
def splitIntoFibonacci(S):
    def isfib(str1,str2,base,ans):
        sumstr=str(int(str1)+int(str2))
        n=len(sumstr)
        if n>len(base):
            return []
        elif n==len(base):
            if int(sumstr)==int(base):
                ans.append(str1)
                ans.append(str2)
                ans.append(base)
                return ans
            else:
                return []
        else:
            if int(sumstr)!=int(base[:n]):
                return []
            else:
                ans.append(str1)
                return isfib(str2,sumstr,base[n:],ans)
    s=0
    for x in range(len(S)):
        if S[x]=='0':
            s+=1
        else:
            break
    if s==len(S):
        return list(S)
    elif s>0:
        str1=0
        base=S[s:]
        ans=[0]*(s-1)
        for x in range((len(S)-s)//2):       
            str2=base[:x+1]
            print(str2,ans)
            if isfib(str1,str2,base[x+1:],ans)==True:
                return ans
    else:
        base=S
        ans=[]
        for x in range(len(S)//3):
            str1=base[:x+1]
            for y in range(len(S)//3):
                str2=base[x+1:x+1+y+1]
                if isfib(str1,str2,base[x+y+2:],ans)==True:
                    return ans
    return []
print(splitIntoFibonacci("112"))
"""

#954. Array of Doubled Pairs
"""
def canReorderDoubled(A):
    A.sort(key=lambda x:abs(x))
    pos=[]
    neg=[]
    for x in A:
        if x>=0:
            if not pos or x!=2*pos[0]:
                pos.append(x)
            else:
                pos.pop(0)
        else:
            if not neg or x!=2*neg[0]:
                neg.append(x)
            else:
                neg.pop(0)
    return not pos and not neg
print(canReorderDoubled(A = [1,2,4,16,8,4]))
"""

#948. Bag of Tokens
"""
def bagOfTokensScore(tokens, P):
"""

#1356. Sort Integers by The Number of 1 Bits
"""
def sortByBits(arr):
    base=[[sum(map(int,bin(x)[2:])),x] for x in arr]
    return [x[1] for x in sorted(base,key=lambda x: [x[0],x[1]])]
print(sortByBits(arr = [1024,512,256,128,64,32,16,8,4,2,1]))
"""

#1073. Adding Two Negabinary Numbers
"""
def addNegabinary(arr1, arr2):  
    dec1=int("".join(map(str,arr1)),2)
    b1=""
    for x in range(len(arr1)-2,-1,-2):
        b1=str(arr1[x])+b1
    b1=b1+"0"
    qua1=int(b1,4)
    dec2=int("".join(map(str,arr2)),2)
    b2=""
    for x in range(len(arr2)-2,-1,-2):
        b2=str(arr2[x])+b2
    b2=b2+"0"
    qua2=int(b2,4)
    base1=dec1-qua1
    base2=dec2-qua2
    ans=base1+base2
    def conv2(n):
        if n == 0:
            return [0]
        out = []
        while n != 0:
            n, rem = divmod(n, -2)
            if rem < 0:
                n += 1
                rem -= -2
            out.append(rem)
        return out[::-1]

    return  conv2(ans)

print(addNegabinary(arr1 = [1,1,1,1,1], arr2 = [1,0,1]))
"""

#1276. Number of Burgers with No Waste of Ingredients
"""
def numOfBurgers(tomatoSlices, cheeseSlices):
    if cheeseSlices==0:
        return [0,0] if tomatoSlices==0 else []
    elif tomatoSlices/cheeseSlices<2 or tomatoSlices/cheeseSlices>4 or tomatoSlices%2==1:
        return []
    else:
        ans=(tomatoSlices-2*cheeseSlices)//2
        return [ans,cheeseSlices-ans]
print(numOfBurgers(16,7))
"""
#1324. Print Words Vertically
"""
import itertools
def printVertically(s):
    return a=[''.join(a).rstrip() for a in itertools.zip_longest(*s.split(),fillvalue=" ")]
"""

#1328. Break a Palindrome
"""
def breakPalindrome(palindrome):
    if len(palindrome)==1:
        return ""
    elif len(set(palindrome))==1:
        if palindrome[0]=="a":
            return palindrome[:-1]+"b"
        else:
            return "a"+palindrome[1:]
    else:
        if len(palindrome)%2==0:
            i=0
            base=list(palindrome)
            while i<len(palindrome):
                if base[i]!="a":
                    base[i]="a"
                    return "".join(base)
                i+=1
        else:
            i=0
            base=list(palindrome)
            while i<len(palindrome):
                if base[i]!="a":
                    print("istime")
                    if i==len(palindrome)//2:
                        return palindrome[:-1]+"b"
                    else:
                        base[i]="a"
                        return "".join(base)
                i+=1
print(breakPalindrome(palindrome = "aabaa"))
"""

#245. Shortest Word Distance III
"""
def shortestWordDistance(words, word1, word2):
    base=[[w,i] for i,w in enumerate(words)]
    if word1==word2:
        tmp=[w[1] for w in base if w[0]==word1]
        return min(map(lambda x,y:x-y,tmp[1:],tmp[:-1]))
    else:
        tmp=[w for w in base if w[0]==word1 or w[0]==word2]
        ans=len(words)
        p=tmp[0]
        for i in range(1,len(tmp)):
            if tmp[i][0]!=p[0]:
                print(tmp[i],p)
                ans=min(ans,tmp[i][1]-p[1])
            p=tmp[i]

        return ans
print(shortestWordDistance(["a","a","c","b"],"a","b"))
"""

#253. Meeting Rooms II
"""
def minMeetingRooms(intervals):
    r=max([x[1] for x in intervals])
    ans=[0]*(r+2)
    for x in intervals:
        ans[x[0]]+=1
        ans[x[1]+1]-=1
    for x in range(1,len(ans)):
        ans[x]+=ans[x-1]
    return max(ans)

print(minMeetingRooms([[7,10],[2,4]]))
"""

#32. Longest Valid Parentheses
"""
def longestValidParentheses(s):
    stack=[]
    ans=0
    tmp=0
    for x in s:
        if x==")":
            if stack:
                tmp=stack.pop()[1]+tmp+2
                ans=max(ans,tmp)
            else:
                tmp=0
        else:
            stack.append([x,tmp])
            if tmp:
                tmp=0
        print(stack)
    return ans
print(longestValidParentheses(s = "((())))()())"))
"""

#42. Trapping Rain Water
"""
def trap(height):
    if len(height)<=2:
        return 0
    l,r=[],[]
    lmax=height[0]
    for x in height[1:]:        
        l.append(lmax)
        lmax=max(lmax,x)
    rmax=height[-1]
    for y in height[:-1][::-1]:
        r.append(rmax)
        rmax=max(rmax,y)
    ans=0
    print(l,r)
    for x in range(1,len(height)-1):
        ans+=max(0,min(r[::-1][x],l[x-1])-height[x])
    return ans
print(trap(height = [0,1,0,2,1,0,1,3,2,1,2,1]))

"""
#360. Sort Transformed Array
"""
def sortTransformedArray(nums, a, b, c):
    import bisect
    y=float(-a/(2*b))
    k=bisect.bisect_left(nums,y)
"""

#692. Top K Frequent Words
"""
def topKFrequent(words, k):
    import collections
    d=collections.Counter(words)
    return [x[0] for x in sorted(d.items(), key=lambda x:[-x[1],x[0]])[:k]]
print(topKFrequent(["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4))
"""

#493. Reverse Pairs
#unsolved

#1180. Count Substrings with Only One Distinct Letter
"""
def countLetters(S):
    import itertools
    base=[len(list(x)) for k,x in itertools.groupby(S)]
    ans=[(x*(x+1))//2 for x in base]
    return sum(ans)
print(countLetters("aaaba"))
"""

#18. 4Sum
"""
def fourSum(nums, target):
    nums.sort()
    N, result = len(nums), []
    for i in range(N):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        for j in range(i+1, N):
            if j > i+1 and nums[j] == nums[j-1]:
                continue
            x = target - nums[i] - nums[j]
            s,e = j+1, N-1
            while s < e:
                if nums[s]+nums[e] == x:
                    result.append([nums[i], nums[j], nums[s], nums[e]])
                    s = s+1
                    while s < e and nums[s] == nums[s-1]:
                        s = s+1
                elif nums[s]+nums[e] < x:
                    s = s+1
                else:
                    e = e-1
    return result
"""
#986. Interval List Intersections
"""
def intervalIntersection(A, B):
    def getsec(a,b):
        if a[1]<b[0] or a[0]>b[1]:
            return None
        return [max(a[0],b[0]),min(a[1],b[1])]
    ans=[]
    i=0
    j=0
    while i<len(A) and j<len(B):
        if getsec(A[i],B[j]):
            ans.append(getsec(A[i],B[j]))
        if A[i][1]<B[j][1]:
            i+=1
        elif A[i][1]>B[j][1]:
            j+=1
        else:
            i+=1
            j+=1
    return ans
print(intervalIntersection(A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]))
"""

#1055. Shortest Way to Form String

#560. Subarray Sum Equals K

#1054. Distant Barcodes
"""
def rearrangeBarcodes(barcodes):
    import collections
    n=len(barcodes)
    d=collections.Counter(barcodes)
    ans=list(d.items())
    ans.sort(key=lambda x:x[1],reverse=True)
    res=[]
    for x in ans:
        res.extend([x[0]]*x[1])
    tmp=[]
    if n%2==0:
        for x in zip(res[:n//2],res[n//2:]):
            tmp.extend(x)
    else:
        for x in zip(res[:n//2],res[n//2+1:]):
            tmp.extend(x)
        tmp.append(res[n//2])
    return tmp
print(rearrangeBarcodes([1,1,2,2,2,3,3]))
"""

#1151. Minimum Swaps to Group All 1's Together
"""
def minSwaps(data):
    n=len(data)
    k=sum(data)
    j=k
    cur=sum(data[:k])
    ans=k-cur
    print(n,k,j,cur,ans)
    for j in range(k,n):
        cur=cur+data[j]-data[j-k]
        ans=min(ans,k-cur)
    return ans
print(minSwaps(data = [1,0,1,0,1,0,0,1,1,0,1]))
"""

#1198. Find Smallest Common Element in All Rows
"""
def smallestCommonElement(mat):
    import collections
    d=collections.defaultdict(int)
    for x in mat:
        for y in x:
            d[y]+=1
    ans=float('inf')
    for x in d.keys():
        if d[x]==len(mat):
            ans=min(ans,x)
    return ans
print(smallestCommonElement(mat = [[1,2,3,4,5],[2,4,5,8,10],[3,5,7,9,11],[1,3,5,7,9]]))
"""

#1190. Reverse Substrings Between Each Pair of Parentheses
"""
def reverseParentheses(s):
    st=['']
    for x in s:
        if x=="(":
            st.append('')
        elif x==")":
            tmp=st.pop()[::-1]
            st[-1]+=tmp
        else:
            st[-1]+=x
    return ''.join(st)
print(reverseParentheses(s = "(u(love)i)"))
"""

#880. Decoded String at Index
"""
def decodeAtIndex(S, K):
    N=0
    for x in range(len(S)):
        if S[x].isdigit():
            N*=int(S[x])
        else:
            N+=1
        if N>=K:
            break
    for back in range(x,-1,-1):
        if S[back].isdigit():
            N/=int(S[back])
            K%=N
        else:
            if K==N or K==0:
                return S[back]
            N-=1
        
print(decodeAtIndex(S = "a2345678999999999999999", K = 1))
"""

#946. Validate Stack Sequences
"""
def validateStackSequences(pushed, popped):
    st=[]
    n=len(popped)
    i,j=0,0
    while i<n:
        while not st or st[-1]!=popped[j] and i<n:
            st.append(pushed[i])
            i+=1
        while st and st[-1]==popped[j]:
            st.pop()
            j+=1
    return not st
print(validateStackSequences([1,2,3,4,5,6,7],[1,2,5,3,6,7,4]))
"""
#907. Sum of Subarray Minimums
"""
def sumSubarrayMins(arr):
    if len(arr)==1:
        return sum(arr)
    st=[]
    n=len(arr)
    ans=0
    last=0
    for x in range(n):
        cur=1
        while st and st[-1][0]>=arr[x]:
            p=st.pop()
            cur+=p[1]
            last-=p[1]*p[0]
        st.append([arr[x],cur])
        if len(st)>=2:
            print("last=",last)
            last+=st[-1][0]*st[-1][1]
            ans+=last
        else:
            last=st[-1][0]*st[-1][1]
            ans+=last
        print(ans)
    return ans
print(sumSubarrayMins([3,1,2,4]))
"""

#15. 3Sum
"""
def threeSum(nums):

    res=[]
    nums.sort()
    if len(nums)<3:
        return []
    for i in range(len(nums)-2):
        if i>0 and nums[i]==nums[i-1]:
            continue
        l,r=i+1,len(nums)-1
        while l<r:
            s=nums[i]+nums[l]+nums[r]
            if s==0:
                res.append([nums[i],nums[l],nums[r]])
                l+=1
                r-=1
                while l<r and nums[l]==nums[l-1]:l+=1
                while l<r and nums[r]==nums[r+1]:r-=1
            elif s<0:
                l+=1
            else:
                r-=1
    return res
print(threeSum(nums = [-1,0,1,2,-1,-4]))
"""

#1590. Make Sum Divisible by P
"""
def minSubarray(nums, p):
    dp={0:-1}
    need=sum(nums)%p
    cur=0
    res=n=len(nums)
    for i,x in enumerate(nums):
        cur=(cur+x)%p
        dp[cur]=i
        if (cur-need)%p in dp:
            res=min(res,i-dp[(cur-need)%p])
    return res if res!=n else -1
print(minSubarray())
"""
#1386. Cinema Seat Allocation
def maxNumberOfFamilies(n, reservedSeats):
    """
    import collections
    d=collections.defaultdict(list)
    for x in reservedSeats:
        d[x[0]].append(x[1])
    ans=0
    occ=len(d.keys())
    ans+=(n-occ)
    print(ans,d)
    for r in d.keys():
        tmp=0
        if set([2,3,4,5])&set(d[r])==set():
            tmp+=1
        if set([5,6,7,8])&set(d[r])==set():
            tmp+=1
        if set([2,3])&set(d[r])!=set() and set([8,9])&set(d[r])!=set():
            if set([4,5,6,7])&set(d[r])==set():
                tmp+=1
        ans+=tmp
    return ans
print(maxNumberOfFamilies( n = 3, reservedSeats = [[1,2],[1,3],[1,8],[2,6],[3,1],[3,10]]))
"""
#1155. Number of Dice Rolls With Target Sum
"""
unsolved
def numRollsToTarget(d, f, target):
    if target>d*f or target<d:
        return 0
"""

#1094. Car Pooling
"""
def carPooling(trips, capacity):
    n=max(x[2] for x in trips)
    p=[[0,0] for _ in range(n)]
    for x in trips:
        p[x[1]-1][0]+=x[0]
        p[x[2]-1][1]+=x[0]
    tmp=0
    for x in p:
        tmp+=x[0]
        tmp-=x[1]
        if tmp>capacity:
            return False
    return True 
print(carPooling(trips = [[2,1,5],[3,3,7]], capacity = 5))
"""

#1155. Number of Dice Rolls With Target Sum
"""
def numRollsToTarget(d, f, target):
    if target<d or target>d*f:
        return 0
    dp=[[0]*d*f for _ in range(d)]
    for r in range(f):
        dp[0][r]=1
    for dice in range(1,d):
        for face in range(dice,(dice+1)*f):
            for b in range(1,f+1):
                if face-b>=0:
                    dp[dice][face]+=dp[dice-1][face-b]  
                else:
                    continue
    return dp[d-1][target-1]%(10**9+7)
print(numRollsToTarget(d = 30, f = 30, target = 500))
"""
#1291. Sequential Digits
"""
def sequentialDigits(low, high):
    ans=[]
    for x in range(2,10):
        for s in range(1,10-x+1):
            tmp=""
            for e in range(x):
                tmp+=str(s)
                s+=1
            ans.append(tmp)
    res=[]
    for x in ans:
        if int(x)>=low and int(x)<=high:
            res.append(int(x))
    return res

print(sequentialDigits(1000,13000))
"""

#1248. Count Number of Nice Subarrays
"""
def numberOfSubarrays(nums, k):
    base=[x for x,a in enumerate(nums) if a%2==1]
    if len(base)<k:
        return 0
    n=len(base)
    base=[-1]+base+[len(nums)]
    ans=0
    print("base=",base)
    for x in range(n-k+1):
        ans+=(base[x+1]-base[x])*(base[x+k+1]-base[x+k])
    return ans
print(numberOfSubarrays(nums = [2,2,2,1,2,2,1,2,2,2], k = 2))
"""

#1010. Pairs of Songs With Total Durations Divisible by 60
"""
def numPairsDivisibleBy60(time):
    import collections
    d=collections.defaultdict(int)
    for x in time:
        d[x%60]+=1
    ans=0
    for s in d.keys():
        if s<30 and s>0 and 60-s in d.keys():
            ans+=d[s]*d[60-s]
    ans+=d[30]*(d[30]-1)//2+d[0]*(d[0]-1)//2
    return ans
print(numPairsDivisibleBy60( time = [60,60,60]))
"""

#974. Subarray Sums Divisible by K
"""
def subarraysDivByK(A, K):
    import collections
    cur=0
    ans=0
    
    d=collections.defaultdict(int)
    for x in A:
        cur+=x
        if cur%K==0:
            ans+=1
        ans+=d[cur%K]
        d[cur%K]+=1
    return ans
print(subarraysDivByK(A = [4,5,0,-2,-3,1], K = 5))
"""

#978. Longest Turbulent Subarray
"""
def maxTurbulenceSize(arr):
    if len(arr)<3:
        return len(arr)
    ans=0
    tmp=2
    for x in range(2,len(arr)):
        if (arr[x]-arr[x-1])*(arr[x-1]-arr[x-2])<0:
            tmp+=1
            ans=max(ans,tmp)
        else:
            tmp=2
    return ans
print(maxTurbulenceSize(arr = [8,1,3,3,3,1,2,1]))
"""
#984. String Without AAA or BBB
"""
def strWithout3a3b(A, B):
    ans=""
    if A==B:
        for x in range(A):
            ans+="ab"
    elif A>B:
        d=A-B
        if d<=B:
            for x in range(d):
                ans+="aab"
            for y in range(B-d):
                ans+="ab"
        else:
            for x in range(B):
                ans+="aab"
            for y in range(d-B):
                ans+="a"
    else:
        d=B-A
        if d<=A:
            for x in range(d):
                ans+="bba"
            for y in range(A-d):
                ans+="ba"
        else:
            for x in range(A):
                ans+="bba"
            for y in range(d-A):
                ans+="b"
    return ans
print(strWithout3a3b(4,1))
"""
#994. Rotting Oranges
"""
def orangesRotting(grid):
    import collections
    n=len(grid)
    m=len(grid[0])
    cnt=0
    rot=collections.deque()
    for i in range(n):
        for j in range(m):
            if grid[i][j]==1:
                cnt+=1
            elif grid[i][j]==2:
                rot.append((i,j,0))
    seen=set()
    while rot:
        y,x,d=rot.popleft()
        dirs={(y-1,x),(y+1,x),(y,x-1),(y,x+1)}
        for y1,x1 in dirs:
            if 0<=y1<n and 0<=x<m and (y1,x1) not in seen and grid[y1][x1]==1:
                seen.add((y1,x1))
                cnt-=1
                if cnt==0:
                    return d+1
                rot.append((y1,x1,d+1))
    return 0 if cnt==0 else -1 
    """

#325. Maximum Size Subarray Sum Equals k
"""
def maxSubArrayLen(nums, k):
    d={0:-1}
    s=0
    ans=0
    for i in range(len(nums)):
        s+=nums[i]
        if s not in d:
            d[s]=i
        if s-k in d:
            ans=max(i-d[s-k],ans)
    print(d)
    return ans
print(maxSubArrayLen(nums = [1, -1, 5, -2, 3], k = 3))
"""

#334. Increasing Triplet Subsequence
"""
def increasingTriplet(nums):
    i,j=float('-inf'),float('-inf')
    for x in nums:
        if i==float('-inf'):
            i=x
        else:
            if j==float('-inf'):
                if x>i:
                    j=x
                elif x<=i:
                    i=x
            else:
                if x>j:
                    return True
                elif x<=j and x>i:
                    j=x
                else:
                    i=x
    return False
print(increasingTriplet(nums = [2,1,5,0,4,6]))
"""

#435. Non-overlapping Intervals
"""
def eraseOverlapIntervals(intervals):
    end, cnt = float('-inf'), 0
	for s, e in sorted(intervals, key=lambda x: x[1]):
		if s >= end: 
			end = e
		else: 
			cnt += 1
	return cnt
print(eraseOverlapIntervals( [[1,2],[2,3],[3,4],[1,3]]))
"""

#123. Best Time to Buy and Sell Stock III
"""
def maxProfit(prices):
    n=len(prices)
    lhs,rhs=[0],[0]

    lo=prices[0]
    lp=0
    for x in range(1,n):
        lo=min(lo,prices[x])
        lp=max(lp,prices[x]-lo)
        lhs.append(lp)

    hi=prices[-1]
    rp=0
    for y in range(n-2,-1,-1):
        hi=max(hi,prices[y])
        rp=max(rp,hi-prices[y])
        rhs.append(rp)
    
    return max(a[0]+a[1] for a in zip(rhs[::-1],lhs))
print(maxProfit(prices = [3,3,5,0,0,3,1,4]))
"""
#407. Trapping Rain Water II
"""
def trapRainWater(heightMap):
    if not heightMap or not heightMap[0]:
        return 0
    import heapq
    m,n=len(heightMap),len(heightMap[0])
    heap=[]
    seen=set()
    for i in range(m):
        for j in range(n):
            if i==0 or j==0 or i==m-1 or j==n-1:
                heapq.heappush(heap,(heightMap[i][j],i,j))
                seen.add((i,j))
    ans=0
    while heap:
        height,i,j=heapq.heappop(heap)
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            xp,yp=i+dx,j+dy
            if 0<xp<m-1 and 0<yp<n-1 and (xp,yp) not in seen:
                ans+=max(0,height-heightMap[xp][yp])
                heapq.heappush(heap,(max(height,heightMap[xp][yp]),xp,yp))
                seen.add((xp,yp))
    return ans
print(trapRainWater([
  [1,4,3,1,3,2],
  [3,2,1,3,2,4],
  [2,3,3,2,3,1]
]
))
"""

#406. Queue Reconstruction by Height
"""
def reconstructQueue(people):
    import bisect
    """

#293. Flip Game
"""
def generatePossibleNextMoves(s):
    n,ans=len(s),[]
    if n<2:
        return ans
    else:
        for x in range(n-1):
            if s[x]==s[x+1] and s[x]=="+":
                ans.append(s[:x]+"--"+s[x+2:])
    return ans
print(generatePossibleNextMoves( s = "++++"))
"""

#150. Evaluate Reverse Polish Notation
"""
def evalRPN(tokens):
    st=[]
    for x in tokens:
        if x[-1].isdigit():
            st.append(int(x))
        else:
            a=st.pop()
            b=st.pop()
            if x=="+":
                cur=a+b
            elif x=="-":
                cur=b-a
            elif x=="*":
                cur=a*b
            else:
                cur=int(float(b)/a)
            st.append(cur)
    return st.pop()
print(evalRPN( ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]))
"""

#335. Self Crossing
"""
def isSelfCrossing(x):
    n=len(x)
    for i in range(3,n):
        if i>=3 and     x[i]>=x[i-2] and x[i-1]<=x[i-3]:
            return True
        if i>=4 and x[i]+x[i-4]>=x[i-2] and x[i-1]==x[i-3]:
            return True
        if i>=5 and x[i-5]>=x[i-3]-x[i-1] and x[i-2]>=x[i-4] and x[i-1]<=x[i-3] and x[i-4]>=x[i-2]-x[i]:
            return True
    return False
print(isSelfCrossing([3,3,3,2,1,1]))
"""

#683. K Empty Slots
"""
def kEmptySlots(bulbs, k):
    import bisect
    l=[]
    for d,b in enumerate(bulbs,1):
        i=bisect.bisect(l,b)
        for n in l[i-(i>0):i+1]:
            if abs(b-n)==k+1:
                return d+1
        l.insert(i,b)
    return -1

print(kEmptySlots( bulbs = [1,2,3], k = 1))
"""

#1582. Special Positions in a Binary Matrix
"""
def numSpecial(mat):
    isin=set()
    imul=set()
    jsin=set()
    jmul=set()
    
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j]==1:
                if i not in imul:
                    if i not in isin:
                        isin.add(i)
                    else:
                        isin.remove(i)
                        imul.add(i)
                if j not in jmul:
                    if j not in jsin:
                        jsin.add(j)
                    else:
                        jsin.remove(j)
                        jmul.add(j)
    ans=0
    for x in isin:
        for y in jsin:
            if mat[x][y]==1:
                ans+=1
    
    return ans 
print(numSpecial(mat = [[0,0,0,0,0],
              [1,0,0,0,0],
              [0,1,0,0,0],
              [0,0,1,0,0],
              [0,0,0,1,1]]))
                """



#723. Candy Crush
"""
def candyCrush(board):
    crush=set()
    while True:
        for i in range(len(board)):
            for j in range(len(board[0])):
                if j>1 and board[i][j] and board[i][j]==board[i][j-1]==board[i][j-2]:
                    crush |= {(i,j),(i,j-1),(i,j-2)}
                if i>1 and board[i][j] and board[i][j]==board[i-1][j]==board[i-2][j]:
                    crush |= {(i,j),(i-1,j),(i-2,j)}
        if not crush:
            break
        for i,j in crush:
            board[i][j]=0

        #drop
        for j in range(len(board[0])):
            idx=len(board)-1
            for i in reversed(range(len(board))):
                if board[i][j]:
                    board[idx][j]=board[i][j]
                    idx-=1
            for i in range(idx+1):
                board[i][j]=0
        return board
print(candyCrush([[110,5,112,113,114],[210,211,5,213,214],[310,311,3,313,314],[410,411,412,5,414],[5,1,512,3,3],[610,4,1,613,614],[710,1,2,713,714],[810,1,2,1,1],[1,1,2,2,2],[4,1,4,4,1014]]
))
"""

#1708. Largest Subarray Length K
"""
def largestSubarray(nums, k):
    if k==len(nums):
        return nums
    idx=0
    for i in range(1,len(nums)-k+1):
        idx=i if nums[i]>nums[idx] else idx
    return nums[idx:idx+k]
print(largestSubarray( nums = [1,4,5,2,3], k = 3))
"""

#239. Sliding Window Maximum
def maxSlidingWindow(nums, k):
    import collections
    dq=collections.deque()
    ans=[]
    if not nums:
        return ans
    if k==0:
        return nums
    for i in range(k):
        while len(dq)!=0:
            if nums[i]>nums[dq[-1]]:
                dq.pop()
            else:
                break
        dq.append(i)
    return dq
print(maxSlidingWindow( nums = [1,2,1,2,3,1,0,-1,-3,5,3,6,7], k = 7))



