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

#956. Tallest Billboard
"""
def tallestBillboard(rods):
    import collections
    dp=dict()
    dp[0]=0
    for r in rods:
        tmp=collections.defaultdict(int)
        for d in dp:
            tmp[d+r]=max(dp[d]+r,tmp[r+d])
            tmp[d]=max(dp[d],tmp[d])
            tmp[d-r]=max(dp[d],tmp[d-r])
        dp=tmp
    return dp[0]
print(tallestBillboard([1,2,3,4,5,6]))
"""

#651. 4 Keys Keyboard
"""
def maxA(N):
    best=[0,1]
    for x in range(2,N+1):
        cur=best[x-1]+1
        for y in range(x-1):
            cur=max(cur,best[y]*(x-y-1))
        best.append(cur)
    return best[N]
print(maxA(12))
"""

#673. Number of Longest Increasing Subsequence
"""
def findNumberOfLIS(nums):
    n=len(nums)
    dp=[[1,1] for _ in range(n)]
    ans=1
    for i,num in enumerate(nums):
        cur=1
        cnt=0
        for j in range(i):
            if num>nums[j]:
                cur=max(cur,dp[j][0]+1)
        for j in range(i):
            if dp[j][0]==cur-1 and nums[j]<num:
                cnt+=dp[j][1]
        dp[i]=[cur,max(cnt,dp[i][1])]
        ans=max(ans,cur)
    return sum(x[1] for x in dp if x[0]==ans)
print(findNumberOfLIS( nums = [2,2,2,2,2]))
"""

#1692. Count Ways to Distribute Candies
"""
def waysToDistribute(n,k):
    
    dp=[[1]*n for _ in range(n)]
    for c in range(2,n):
        for b in range(1,min(c,k)):
            dp[c][b]=dp[c-1][b-1]+dp[c-1][b]*(b+1)
    return dp[n-1][k-1]
print(waysToDistribute(7,4))
"""

#91. Decode Ways
"""
def numDecodings(s):
    dp=[0]*(len(s)+1)
    dp[-1]=1
    dp[-2]=0 if s[-1]=='0' else 1
    for x in range(len(s)-2,-1,-1):
        if 0<int(s[x]):
            dp[x]=dp[x+1]
        if 10<=int(s[x]+s[x+1])<=26:
            dp[x]+=dp[x+2]
    print(dp)
    return dp[0]
print(numDecodings("1"))
"""

#1345. Jump Game IV
"""
def minJumps(arr):
    import collections
    if len(arr)==1:
        return 0
    q=collections.deque()
    seen=set()
    m=collections.defaultdict(list)
    d=len(arr)-1
    mj=0

    for i,x in enumerate(arr):
        m[x].append(i)
    
    q.append(0)
    seen.add(0)

    while q:
        size=len(q)
        for i in range(size):
            pos=q.popleft()
            if pos==d:
                return mj
"""
"""
def countArrangement(N):
    d={}
    def helper(i,x):
        if i==1:
            return 1
        key=(i,x)
        if key in d:
            return d[key]
        ans=0
        for j in range(len(x)):
            if x[j]%i==0 or i%x[j]==0:
                ans+=helper(i-1,x[:j]+x[j+1:])
        d[key]=ans
        return ans
    return helper(N,tuple(range(1,N+1)))
print(countArrangement(11))
"""

#256. Paint House
"""
def minCost(costs):
    n = len(costs)
    dp1 = [0] * (n + 1)
    dp2 = [0] * (n + 1)
    dp3 = [0] * (n + 1)
    for idx in range(1, n + 1):
        dp1[idx] = min(dp2[idx - 1], dp3[idx - 1]) + costs[idx - 1][0]
        dp2[idx] = min(dp1[idx - 1], dp3[idx - 1]) + costs[idx - 1][1]
        dp3[idx] = min(dp2[idx - 1], dp1[idx - 1]) + costs[idx - 1][2]
    return min(dp1[-1], dp2[-1], dp3[-1])
print(minCost(costs = [[17,2,17],[16,16,5],[14,3,19]]))
"""
#1049. Last Stone Weight II
"""
def lastStoneWeightII(stones):
    dp = {0}
        sumA = sum(stones)
        for a in stones:
            dp |= {a + i for i in dp}
        return min(abs(sumA - 2 * i) for i in dp)

print(lastStoneWeightII([1,1,2,4,7,8]))
"""
#1024. Video Stitching
"""
def videoStitching(clips, T):
    ans = 0
    clips.sort()
    start, end = 0, 0
    idx = 0
    while start <= end:
        ans += 1
        newstart, newend = end + 1, end
        while idx < len(clips) and start <= clips[idx][0] <= end:
            newend = max(clips[idx][1], newend)
            if newend >= T:
                return ans
            idx += 1
        start, end = newstart, newend
    return -1

print(videoStitching(clips = [[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]], T = 10))"""


#221. Maximal Square
"""
def maximalSquare(matrix):
    if len(matrix) == 0 or len(matrix[0]) == 0:
        return -1
    m, n = len(matrix), len(matrix[0])
    dp = [[0 if matrix[i][j] == '0' else 1 for j in range(n)] for i in range(m)]
    print(dp)
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == '1':
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
            else:
                dp[i][j] = 0
    res = max(max(r) for r in dp)
    print(dp)
    return res ** 2
print(maximalSquare( matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]))
"""


#1277. Count Square Submatrices with All Ones
"""
def countSquares(matrix):
        m, n = len(matrix), len(matrix[0])
        if m == 1:
            return sum(matrix[0])
        if n == 1:
            return sum(map(lambda x:x[0], matrix))
        ans = sum(matrix[0]) + sum(map(lambda x:x[0], matrix)) - matrix[0][0]
        dp = [[0 if matrix[i][j] == 0 else 1 for j in range(n)] for i in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 1:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    ans += dp[i][j]
                else:
                    dp[i][j] = 0
        return ans
print(countSquares(matrix =
[
  [0,1,1,1],
  [1,1,1,1],
  [0,1,1,1]
]))
"""

#750. Number Of Corner Rectangles
"""
def countCornerRectangles(grid):
    if len(grid) == 1 or len(grid[0]) == 1:
        return 0
    ans = 0
    ls = []
    for idx in range(len(grid)):
        curset = set()
        for i, x in enumerate(grid[idx]):
            if x == 1:
                curset.add(i)
        if idx != 0:
            for s in ls:
                num = len(s & curset)
                if num >= 2:
                    ans += num * (num -1) // 2
        ls.append(curset)
    return ans
print(countCornerRectangles(grid = 
[[1, 1, 1],
 [1, 1, 1],
 [1, 1, 1]]))
"""


