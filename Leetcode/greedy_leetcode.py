#944. Delete Columns to Make Sorted
"""
def minDeletionSize(A):
    import functools
    base=[functools.reduce(lambda x,y:x+y,[m[x] for m in A]) for x in range(len(A[0]))]
    c=[1 for a in base if list(a)==sorted(a)]
    return len(A[0])-sum(c)


print(minDeletionSize( A = ["zyx","wvu","tsr"]))
"""

#1288. Remove Covered Intervals
"""
def removeCoveredIntervals(intervals):
    intervals.sort(key=lambda x: x[0])

    tmp=intervals[0]
    cnt=1
    for i in range(1,len(intervals)):
        cur=intervals[i]
        if cur[1]>tmp[1]:
            cnt+=1
        tmp[0]=max(tmp[0],cur[0])
        tmp[1]=max(tmp[1],cur[1])
    return cnt
print(removeCoveredIntervals(intervals = [[1,2],[3,5],[5,8],[6,8]]))
"""

#1253. Reconstruct a 2-Row Binary Matrix
"""
def reconstructMatrix(upper, lower, colsum):
    import collections
    c = collections.Counter(colsum)
    if c[2] > min(upper, lower) or sum(colsum) != upper + lower:
        return []
    n = len(colsum)
    ans = [[0]*n for _ in range(2)]
    upmore = upper - c[2]
    lomore = lower - c[2]
    for i, x in enumerate(colsum):
        if x == 2:
            ans[0][i] = 1
            ans[1][i] = 1
        elif x == 1:
            if upmore > 0:
                ans[0][i] = 1
                upmore -= 1
            else:
                ans[1][i] = 1
        else:
            continue
    return ans
print(reconstructMatrix(upper = 5, lower = 5, colsum = [2,1,2,0,1,0,1,2,0,1]))
"""

#765. Couples Holding Hands
"""
def minSwapsCouples(row):
    res = [r//2 for r in row]
    ans = 0
    for i in range(0, len(row), 2):
        j = res.index(res[i], i + 1)
        if j != i + 1:
            res[j] = res[i + 1]
            ans += 1      
    return ans
print(minSwapsCouples(row = [0, 2, 1, 3]))
"""

#1058. Minimize Rounding Error to Meet Target
"""
def minimizeError(prices, target):
    import math
    num = [round(float(x)-int(float(x)),3) for x in prices if float(x) != int(float(x))]
    f = target - sum(math.floor(float(x)) for x in prices)
    if f == len(num):
        res = len(num) - sum(num)
        return '{:.3f}'.format(res)
    elif f == 0:
        res = sum(num)
        return '{:.3f}'.format(res)
    else:
        ans = 0
        num.sort(reverse = True)
        for idx in range(f):
            ans += 1 - num[idx]
        for idx in range(f, len(num)):
            ans += num[idx]
        res = round(ans,3)
        return '{:.3f}'.format(res)
print(minimizeError(prices = ["1.500","2.500","3.500"], target = 8))
"""
#1536. Minimum Swaps to Arrange a Binary Grid
"""
def minSwaps(grid):
    import itertools
    res = []
    for idx in range(len(grid)):
        cur = [[g, len(list(k))] for g,k in itertools.groupby(grid[idx])]
        i = cur[-1][1] if cur[-1][0] == 0 else 0
        res.append(i)
    for i, x in enumerate(sorted(res)):
        if i > x:
            return -1
    ans = 0
    for k in range(len(grid)-1,0,-1): 
        for i, v in enumerate(res): 
            if v >= k:
                ans += i
                res.pop(i)
    return ans 
print(minSwaps(grid = [[0,0,1],[1,1,0],[1,0,0]]))
"""

#1594. Maximum Non Negative Product in a Matrix
def maxProductPath(A):
    """
    import functools
    m, n = len(grid), len(grid[0])
    if m == 1 and n == 1:
        ans = grid[0][0]
        return ans if ans >= 0 else -1
    elif m == 1:
        ans = functools.reduce(lambda x, y : x * y, grid[0])
        return ans if ans >= 0 else -1
    elif n == 1:
        ans = functools.reduce(lambda x, y : sum(x) * sum(y), grid)
        return ans if ans >= 0 else -1
    else:
        tmp = [[["#", "#"] for i in range(n)] for _ in range(m)]
        if grid[0][0] == 0:
            tmp[0][0] = [0, 0]
        elif grid[0][0] < 0:
            tmp[0][0] = ["#", grid[0][0]]
        else:
            tmp[0][0] = [grid[0][0], "#"]
        s = grid[0][0]
        for i in range(1, n):
            s *= grid[0][i]
            if s == 0:
                tmp[0][i] = [0, 0]
            elif s > 0:
                tmp[0][i] = [s, "#"]
            else:
                tmp[0][i] = ["#", s]
        s = grid[0][0]
        for j in range(1, m):
            s *= grid[j][0]
            if s == 0:
                tmp[j][0] = [0, 0]
            elif s > 0:
                tmp[j][0] = [s, "#"]
            else:
                tmp[j][0] = ["#", s]
        for i in range(1, m):
            for j in range(1, n):
                cur = grid[i][j]
                if cur == 0:
                    tmp[i][j] = [0, 0]
                elif cur > 0:
                    if tmp[i-1][j][0] == "#" and tmp[i][j-1][0] == "#":
                        pos = "#"
                    elif tmp[i-1][j][0] == "#":
                        pos = cur * tmp[i][j-1][0]
                    elif tmp[i][j-1][0] == "#":
                        pos = cur * tmp[i-1][j][0]
                    else:
                        pos = max(cur * tmp[i-1][j][0], cur * tmp[i][j-1][0])
                    
                    if tmp[i-1][j][1] == "#" and tmp[i][j-1][1] == "#":
                        neg = "#"
                    elif tmp[i-1][j][1] =="#":
                        neg = cur * tmp[i][j-1][1]
                    elif tmp[i][j-1][1] == "#":
                        neg = cur * tmp[i-1][j][1]
                    else:
                        neg = min(cur * tmp[i-1][j][1], cur * tmp[i][j-1][1])
                    tmp[i][j] = [pos, neg]
                else:
                    if tmp[i-1][j][0] == "#" and tmp[i][j-1][0] == "#":
                        neg = "#"
                    elif tmp[i-1][j][0] == "#":
                        neg = cur * tmp[i][j-1][0]
                    elif tmp[i][j-1][0] == "#":
                        neg = cur * tmp[i-1][j][0]
                    else:
                        neg = min(cur * tmp[i-1][j][0], cur * tmp[i][j-1][0])
                    
                    if tmp[i-1][j][1] == "#" and tmp[i][j-1][1] == "#":
                        pos = "#"
                    elif tmp[i-1][j][1] =="#":
                        pos = cur * tmp[i][j-1][1]
                    elif tmp[i][j-1][1] == "#":
                        pos = cur * tmp[i-1][j][1]
                    else:
                        pos = max(cur * tmp[i-1][j][1], cur * tmp[i][j-1][1])
                    tmp[i][j] = [pos, neg]
        return tmp[-1][-1][0] if tmp[-1][-1][0] != "#" else -1
    m, n = len(A), len(A[0])
    Max = [[0] * n for _ in range(m)]
    Min = [[0] * n for _ in range(m)]
    Max[0][0] = A[0][0]
    Min[0][0] = A[0][0]
    for j in range(1, n):
        Max[0][j] = Max[0][j - 1] * A[0][j]
        Min[0][j] = Min[0][j - 1] * A[0][j]

    for i in range(1, m):
        Max[i][0] = Max[i - 1][0] * A[i][0]
        Min[i][0] = Min[i - 1][0] * A[i][0]
    for i in range(1, m):
        for j in range(1, n):
            if A[i][j] > 0:
                Max[i][j] = max(Max[i - 1][j], Max[i][j - 1]) * A[i][j]
                Min[i][j] = min(Min[i - 1][j], Min[i][j - 1]) * A[i][j]
            else:
                Max[i][j] = min(Min[i - 1][j], Min[i][j - 1]) * A[i][j]
                Min[i][j] = max(Max[i - 1][j], Max[i][j - 1]) * A[i][j]
    print(Max)
    return Max[-1][-1] % int(1e9 + 7) if Max[-1][-1] >= 0 else -1
print(maxProductPath(A = [[-1,-2,-3],
               [-2,-3,-3],
               [-3,-3,-2]]))
"""

#995. Minimum Number of K Consecutive Bit Flips
"""
def minKBitFlips(A, K):
    n = len(A)
    f = [0] * (n + 1)
    ans = 0
    cur = 0 
    for x in range(n - K + 1):
        cur += f[x]
        if (A[x] + cur) % 2 == 1:
            continue
        else:
            ans += 1
            cur += 1
            f[x + K] -= 1
    for idx in range(x + 1, n):
        cur += f[idx]
        if (A[idx] + cur) % 2 == 0:
            return -1
    return ans
print(minKBitFlips(A = [0,0,0,1,0,1,1,0], K = 3))
"""

#1505. Minimum Possible Integer After at Most K Adjacent Swaps On Digits

"""
def minInteger(N, k):
    import collections
    ans = []
    num = list(N)
    n = len(num)
    while k > 0 and num:
        l = min(len(num), k + 1)
        cur = collections.Counter(num[ : l])
        t = min(cur.keys())
        idx = num.index(t)
        ans.append(num.pop(idx))
        k -= idx
    a = "".join(ans) + "".join(num)
    return a
"""

#321. Create Maximum Number
"""
def maxNumber(nums1, nums2, k):
    def getmax(nums, t):
        ans = []
        n = len(nums)
        for idx in range(n):
            while ans and ans[-1] < nums[idx] and len(ans) + n - idx > t:
                ans.pop()
            if len(ans) < t:
                ans.append(nums[idx])
        return ans
    def merge(nums1, nums2):
        ans = []
        while nums1 or nums2:
            if nums1 > nums2:
                ans.append(nums1[0])
                nums1 = nums1[1:]
            else:
                ans.append(nums2[0])
                nums2 = nums2[1:]
        return ans

    l1, l2 = len(nums1), len(nums2)
    ans = []
    for idx in range(max(0, k - l2), min(k, l1) + 1):
        tmp = merge(getmax(nums1, idx), getmax(nums2, k - idx))
        ans = max(tmp, ans)
    return ans
print(maxNumber(nums1 = [3, 4, 6, 5],
nums2 = [9, 1, 2, 5, 8, 3],
k = 5))"""

#1405. Longest Happy String
"""
def longestDiverseString(a, b, c):
    import math
    tmp = [[a, "a"], [b, "b"], [c, "c"]]
    tmp.sort(key = lambda x: x[0], reverse = True)
    i, j, k = tmp[0], tmp[1], tmp[2]
    if i[0] > (j[0] + k[0]) * 2 + 2:
        res = j[0] * j[1] + k[1] * k[0]
        ans = ""
        for r in res:
            ans += i[1] * 2 + r
            ans += i[1] * 2
    else:
        fence = [""] * math.ceil(i[0] / 2)
        if j[0] >= len(fence):
            more = j[0] - len(fence)
            for idx in range(more):
                fence[idx] += j[1]*2
            for idx in range(more, len(fence)):
                fence[idx] += j[1]
            more = k[0] - len(fence)
            print(more)
            if more > 0:
                for idx in range(more):
                    fence[idx] += k[1] * 2
                for idx in range(more, len(fence)):
                    fence[idx] += k[1]
            else:
                for idx in range(k[0]):
                    fence[idx] += k[1]
            print(fence)
        else:
            for idx in range(j[0]):
                fence[idx] += j[1]
            for idx in range(1, k[0] + 1):
                fence[-idx] += k[1]
        ans = ""
        ifence = [i[1] * 2] * len(fence)
        if i[0] % 2:
            ifence[-1] = i[1]
        for idx in range(len(fence)):
            ans += ifence[idx] + fence[idx]
    return ans
print(longestDiverseString(11,13,7))
"""

#1354. Construct Target Array With Multiple Sums
"""
def isPossible(target):
    import heapq
    tmp = [-x for x in target]   
    tmp.sort()
    heapq.heapify(tmp)   
    s = sum(tmp)
    p = heapq.heappop(tmp)
    while p != -1:
        if 2 * p - s> -1:
            return False
        cur = -(p % (s - p)) if p % (s - p) != 0 else s - p 
        print("p=", p, "s=", s, "cur=", cur)
        heapq.heappush(tmp, cur)
        s += (cur - p)
        p = heapq.heappop(tmp)
    return True
print(isPossible([1,1,1,12]))
"""
        





            

