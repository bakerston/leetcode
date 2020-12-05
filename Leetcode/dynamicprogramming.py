#1105. Filling Bookcase Shelves
"""
def minHeightShelves(books, shelf_width):
    n=len(books)
    dp=[float('inf')]*(n+1)
    print(dp)
    dp[0]=0
    for i in range(1,n+1):
        max_width=shelf_width
        max_height=0
        j=i-1
        while j>=0 and max_width>=books[j][0]:
            max_width-=books[j][0]
            max_height=max(max_height,books[j][1])
            dp[i]=min(dp[i],dp[j]+max_height)
            j-=1
    print(dp)
    return dp[n]
print(minHeightShelves(books = [[1,1],[2,3],[2,3],[1,1],[1,1],[1,1],[1,2]], shelf_width = 4))
"""

#1631. Path With Minimum Effort
"""
def minimumEffortPath(heights):
    r=len(heights)
    c=len(heights[0])
    if r==1 and c==1:
        return 0
    elif r==1:
        return max([abs(x-y) for x,y in zip(heights[0][1:],heights[0][:-1])])
    elif c==1:
        return max([abs(x[0]-y[0]) for x,y in zip(heights[1:],heights[:-1])])
    else:
        dp=[[float('inf') for x in range(c)] for y in range(r)]
        dp[0][0]==0
        def findpath(x,y):
            if x==0:
                if y==c-1:
                    dp[x][y]=min(dp[x][y],max(dp[x][y-1],abs(heights[x][y-1]-heights[x][y])))
                elif y==0:
                    dp[x][y]=0
                else:
                    dp[x][y]=min(dp[x][y], max(dp[x][y-1],abs(heights[x][y-1]-heights[x][y])) ,max(dp[x+1][y],abs(heights[x+1][y]-heights[x][y]))) 
            elif x==r-1:
                if y==0:
                    dp[x][y]=min(dp[x][y], max(dp[x-1][y],abs(heights[x-1][y]-heights[x][y])))             
                else:
                    dp[x][y]=min(dp[x][y],max(dp[x-1][y],abs(heights[x-1][y]-heights[x][y])),max(dp[x][y-1],abs(heights[x][y-1]-heights[x][y])))
            else:
                if y==0:
                    dp[x][y]=min(dp[x][y],max(dp[x-1][y],abs(heights[x-1][y]-heights[x][y])),max(dp[x][y+1],abs(heights[x][y+1]-heights[x][y])))
                elif y==r-1:
                    dp[x][y]=min(dp[x][y],max(dp[x-1][y],abs(heights[x-1][y]-heights[x][y])),max(dp[x][y-1],abs(heights[x][y-1]-heights[x][y])))
                else:
                    dp[x][y]=min(dp[x][y],max(dp[x-1][y],abs(heights[x-1][y]-heights[x][y])),max(dp[x][y+1],abs(heights[x][y+1]-heights[x][y])),max(dp[x+1][y],abs(heights[x+1][y]-heights[x][y])),max(dp[x][y-1],abs(heights[x][y+-1]-heights[x][y])))
    for x in range(r):
        for y in range(c):
            findpath(x,y)
            print(x,y,dp[x][y])
    return dp[r-1][c-1]
print(minimumEffortPath(
[[1,2,1,1,1],[1,2,1,2,1],[1,2,1,2,1],[1,2,1,2,1],[1,1,1,2,1]]))
"""

#1546. Maximum Number of Non-Overlapping Subarrays With Sum Equals Target
"""
def maxNonOverlapping(nums, target):
    import collections
    presum=set()
    s=0
    presum.add(0)
    ind=collections.defaultdict(int)
    ans=0
    for x in range(len(nums)):
        s+=nums[x]
        if s-target in presum:
            ind[s]=max(ind[s],ind[s-target]+1,ans)
            ans=max(ind[s],ans)
        else: 
            ind[s]=max(ind[s],ans)
        presum.add(s)
    return ans
print(maxNonOverlapping( nums = [-1,3,5,1,4,2,-9], target = 6))
"""

#1553. Minimum Number of Days to Eat N Orangesm
"""
def minDays(n):
    dp=[0]*(n+1)
    if n <= 1:
        return n
    else:
        if dp[n]!=0:
            return dp
        else:
            return 1 + min(n % 2 + minDays(n // 2), n % 3 + minDays(n // 3)) 

print(minDays(56))
"""

#1043. Partition Array for Maximum Sum
"""
def maxSumAfterPartitioning(A, k):
        N = len(A)
        dp = [0] * (N + 1)
        for i in range(N):
            curMax = 0
            for k in range(1, min(k, i + 1) + 1):
                curMax = max(curMax, A[i - k + 1])
                dp[i] = max(dp[i], dp[i - k] + curMax * k)
        return dp[N ]
print(maxSumAfterPartitioning([1],1))
"""




#1567. Maximum Length of Subarray With Positive Product
"""
def getMaxLen(nums):
    
    n=len(nums)
    dp=[[0]*2 for x in range(n)]

    if nums[0]>0:
        dp[0][0]=1
    if nums[0]<0:
        dp[0][1]=1
    
    res=dp[0][0]
    for i in range(1,n):
        cur=nums[i]

        if cur>0:
            dp[i][0]=dp[i-1][0]+1
            if dp[i-1][1]>0:
                dp[i][1]= max(dp[i][1],dp[i-1][1]+1)
        if cur<0:
            dp[i][1]=dp[i-1][0]+1
            if dp[i-1][1]>0:
                dp[i][0]=max(dp[i][0],dp[i-1][1]+1)
        res=max(res,dp[i][0])
    return res
print(getMaxLen(nums = [0,1,-2,-3,-4]))
"""

#72. Edit Distance
"""
def minDistance(word1, word2):
    m=len(word1)
    n=len(word2)
    t=[[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        t[i][0]=i
    for j in range(n+1):
        t[0][j]=j
    for i in range(1,m+1):
        for j in range(1,n+1):
            if word1[i-1]==word2[j-1]:
                t[i][j]=t[i-1][j-1]
            else:
                t[i][j]=1+min(t[i][j-1],t[i-1][j],t[i-1][j-1])
    return t[-1][-1]
print(minDistance(word1 = "horse", word2 = "ros"))
"""

#871. Minimum Number of Refueling Stops
"""
def minRefuelStops(target, startFuel, stations):
    dp=[startFuel]+[0]*len(stations)
    for i in range(len(stations)):
        for d in range(len(stations)+1)[::-1]:
            if dp[d] >= stations[i][0]:
                dp[d+1]=max(dp[d+1],dp[d]+target[d][1])
    for i,x in enumerate(dp):
        if x>=target:
            return i
    return -1

print(minRefuelStops(target = 100, startFuel = 1, stations = [[10,100]]))
"""

#1048. Longest String Chain
"""
def longestStrChain(words):
    words.sort(key=len)
    d={}
    for x in words:
        tmp=[x[:i]+x[i+1:] for i in range(len(x))]
        d[x]=max([d.get(w,0) for w in tmp])+1
    return max(d.values())
print(longestStrChain(words = ["a","b","ba","bca","bda","bdca"]))
"""