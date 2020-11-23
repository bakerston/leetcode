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





