#1523. Count Odd Numbers in an Interval Range
"""
def countOdds(low, high):
    return (high-low)//2 if (low%2==0 and high%2==0) else (high-low)//2+1
print(countOdds(1,10))
"""

#1624. Largest Substring Between Two Equal Characters
"""
def maxLengthBetweenEqualCharacters(s):
    adic={}
    alen=len(s)
    for x in range(alen):
        if s[x] not in adic.keys():
            adic[s[x]]=[x]
        else:
            adic[s[x]].append(x)
    return max(map(lambda x: x[-1]-x[0]-1, list(adic.values())))
print(maxLengthBetweenEqualCharacters("cabbac"))
"""
#1566. Detect Pattern of Length M Repeated K or More Times
"""
def containsPattern(arr, m, k):
    if len(arr)==1:
        return m==1 and k==1
    else:
        alen=m*k
        for i in range(len(arr)-alen+1):
            base=[tuple(arr[i+x*m:i+m+x*m]) for x in range(k)]
            if len(set(base))==1:
                return True
        return False
print(containsPattern(arr = [2,2,2,2], m = 2, k = 3))
"""

#1561. Maximum Number of Coins You Can Get
"""
def maxCoins(piles):
    
    alen=int(len(piles)/3)
    base=sorted(piles)[::-1]
    ans=[base[1+2*x] for x in range(alen)]
    return sum(ans)
    return sum(sorted(piles)[len(piles) // 3::2])
print(maxCoins(piles = [2,4,1,2,7,8]))
"""

#1496. Path Crossing
"""
def isPathCrossing(path):
    pdic={"N":[0,1],"E":[1,0],"S":[0,-1],"W":[-1,0]}
    dset={(0,0)}
    tmp=[0,0]
    for x in list(path):
        tmp=tuple([x+y for x,y in zip(tmp,pdic[x])])
        if tmp in dset:
            return True
        else:
            dset.add(tmp)
    return False
print(isPathCrossing(path = "NESWW"))
"""
#1232. Check If It Is a Straight Line
"""
fk whoever made this trash question.
"""

#1189. Maximum Number of Balloons
"""
def maxNumberOfBalloons(text):
    import collections
    base=collections.Counter(text)
    return min(base["b"],base["a"],base["o"]//2,base["n"],base["l"]//2)
print(maxNumberOfBalloons( text = "leetcode"))"""
#1175. Prime Arrangements
"""
def numPrimeArrangements(n):
    import math
    def func_get_prime(n):
        return len([ p for p in range(2, n) if 0 not in [ p%i for i in range(2,int(math.sqrt(p))+1)]])
    p=func_get_prime(n+1)
    np=n-p
    return math.factorial(np)*math.factorial(p)%(10**9+7)

print(numPrimeArrangements(100))
"""

#1657. Determine if Two Strings Are Close
"""
def closeStrings(word1, word2):
    import collections
    base1=collections.Counter(word1)
    base2=collections.Counter(word2)
    return set(base1.keys())==set(base2.keys()) and sorted(base1.values())==sorted(base2.values())
print(closeStrings(word1 = "cabbba", word2 = "aabbss"))
"""

#1604. Alert Using Same Key-Card Three or More Times in a One Hour Period
"""
def alertNames(keyName, keyTime):
    dic={}
    for x in range(len(keyName)):
        if keyName[x] not in dic.keys():
            dic[keyName[x]]=[keyTime[x]]
        else:
            dic[keyName[x]].append(keyTime[x])
    def conv_time(t):
        return int(t[:2])*60+int(t[3:])
    def lt_onehour(alist):
        if len(alist)<=2:
            return False
        else:
            for i in range(len(alist)-2):
                if alist[i+2]-alist[i]<=60:
                    return True
        return False
    ans=[]
    for x in dic.keys():
        tmp=sorted([conv_time(x) for x in list(dic[x])])
        if lt_onehour(tmp)==True:
            ans.append(x)
    return sorted(ans)
print(alertNames(  ["leslie","leslie","leslie","clare","clare","clare","clare"],
["13:00","13:20","14:00","18:00","18:51","19:30","19:49"]))
"""

#1222. Queens That Can Attack the King
"""
def queensAttacktheKing(queens,king):
    from collections import defaultdict
    a=defaultdict(lambda: 8)
    for x in queens:
        if x[0]==king[0] and x[1]>king[1]:
            a["N"]=min(a["N"],x[1]-king[1])
        elif x[0]==king[0] and x[1]<king[1]:
            a["S"]=min(a["S"],king[1]-x[1])
        elif x[1]==king[1] and x[0]>king[0]:
            a["E"]=min(a["E"],x[0]-king[0]) 
        elif x[1]==king[1] and x[0]<king[0]:
            a["W"]=min(a["W"],king[0]-x[0])
        elif x[1]-king[1]==x[0]-king[0] and x[1]>king[1]:
            a["NE"]=min(a["NE"],x[0]-king[0])  
        elif x[1]-king[1]==x[0]-king[0] and x[1]<king[1]:
            a["SW"]=min(a["SW"],king[0]-x[0])  
        elif x[1]-king[1]==-x[0]+king[0] and x[1]>king[1]:
            a["NW"]=min(a["NW"],x[1]-king[1])
        elif x[1]-king[1]==-x[0]+king[0] and x[1]<king[1]:
            a["SE"]=min(a["SE"],x[0]-king[0])
        else:
            continue
    ans=[]    
    for x in a.keys():
        tmp=[king[0],king[1]]
        if a[x]<8:
            if x=="N":
                tmp[1]+=a[x]
                ans.append(tmp)
            elif x=="S":
                tmp[1]-=a[x]
                ans.append(tmp)
            elif x=="W":
                tmp[0]-=a[x]
                ans.append(tmp)
            elif x=="E":
                tmp[0]+=a[x]
                ans.append(tmp)    
            elif x=="NE":
                tmp[0]+=a[x]
                tmp[1]+=a[x]
                ans.append(tmp)
            elif x=="NW":
                tmp[0]-=a[x]
                tmp[1]+=a[x]
                ans.append(tmp)
            elif x=="SE":
                tmp[0]+=a[x]
                tmp[1]-=a[x]
                ans.append(tmp)
            elif x=="SW":
                tmp[0]-=a[x]
                tmp[1]-=a[x]
                ans.append(tmp)
            else:
                continue    
    return ans
print(queensAttacktheKing( queens = [[5,6],[7,7],[2,1],[0,7],[1,6],[5,1],[3,7],[0,3],[4,0],[1,2],[6,3],[5,0],[0,4],[2,2],[1,1],[6,4],[5,4],[0,0],[2,6],[4,5],[5,2],[1,4],[7,5],[2,3],[0,5],[4,2],[1,0],[2,7],[0,1],[4,6],[6,1],[0,6],[4,3],[1,7]], king = [3,4]))
"""

#1551. Minimum Operations to Make Array Equal
"""
def minOperations(n):
    return int((n+n%2)*(n//2)/2)
print(minOperations(3))
"""

#1529. Bulb Switcher IV
"""
def minFlips(target):
    from itertools import groupby
    cur=[x for x,k in groupby(target)]
    return len(cur) if cur[0]=='1' else len(cur)-1
print(minFlips("001011101"))
"""

#1658. Minimum Operations to Reduce X to Zero
"""
def minOperations(nums,x):
    alen=len(nums)
    l,r=0,1
    tmp=nums[0]
    ans=alen+1
    while r<=2*alen:
        if tmp<x:
            tmp+=nums[r%alen]
            r+=1
            if r>2*alen:
                break         
        elif tmp>x:
            tmp-=nums[l%alen]
            l+=1            
            if l==r:
                r+=1
                tmp+=nums[l%alen]
        else:
            print(l,r,tmp)
            if l==0 or r==alen-1 or l%alen>r%alen:
                ans=min(ans,r-l)
            tmp-=nums[l%alen]
            tmp+=nums[r%alen]
            r+=1
            l+=1    
    return ans if ans<alen+1 else -1
print(minOperations( [5297,4630,701,9863,2861,3825,380,5534,5721,1410,4143,1619,8122,307,9955,7855,3792,5633,4795,8228,2877,874,757,7778,7967,4882,9597,1689,1528,4328,3148,506,6477,3377,6336,7900,6586,155,7750,3021,9345,1639,8983,7919,461,1267,1535,3133,387,3592,2092,5570,6787,3247,7839,6338,1248,6080,7079,4716,7284,8006,7622,7020,6272,4781,3194,2734,55,5882,651,4211,9397,2726,6025,4046,5782,1090,9789,7335,1987,4487,7947,2184,2249,444,8593,9766,6769,4713,5136,2784,6114,7924,58,2204,5757,3620,292,116,4257,2052,4496,8949,8172,7695,1444,7100,6656,7177,1349,7316,6173,3363,7371,6028,7229,3668,5406,810,1332,4158,7008,6500,1115,8795,4321,8533,3842,5639,2736,2628,8615,1961,3713,1573,395,8582,4308,4543,7346,3147,9147,3309,5910,2809,4847,9596,9582,2767,5531,7120,5330,8681,1620,6028,6259,4890,7866,5187,8132,4238,1890,1776,4913,5303,7976,9562,4878,1130,7523,2995,3798,4033,5202,5688,5078,5576,8131,5040,3886,2046,4863,3740,6204,5579,3844,3840,4912,1540,8886,6988,3459,1174,5103,1332,103,685,7350,1580,6855,8102,3033,406,4169,9001,484,940,5074,1083,8392,971,2178,9617,4840,2680,2338,6927,2710,766,9182,2305,4106,1042,6120,1035,9759,2154,227,3966,4079,1329,4100,4752,5529,1830,589,3932,9700,7331,152,6110,6888,4389,8814,4221,2777,7559,1393,1069,8690,6360,3534,2978,971,6984,5797,778,3261,2328,9258,5508,3896,62,9599,3361,1394,7634,2616,3752,492,890,3951,3957,9903,303,5511,8158,6443,1097,4581,2007,2691,9852,1376,8718,7361,1763,9273,1904,4709,321,2513,7193,7524,1606,4497,7987,2557,6676,5254,8380,8815,3032,4659,7876,8553,4554,584,3946,8902,4052,5665,910,6767,8791,769,6601,4082,6349,2193,6529,7929,6790,8797,2122,5641,5481,8955,536,8851,7945,2089,8317,3356,151,9295,6619,8422,5584,1286,342,8060,9912,7803,6322,9289,4024,8287,6400,3152,8858,1040,8671,4850,2369,3471,6244,9655,1689,6277,1889,6372,2356,3191,1679,1547,3771,7100,2884,7052,8577,692,9321,2755,4748,8861,6255,3324,3446,7111,3909,9860,7079,4585,3459,9142,9475,4612,7990,2226,5714,5882,2566,8240,4078,7992,7313,9980,8246,1418,7782,691,1507,1825,4466,2757,5950,163,8669,82,5877,5889,9539,3617,6002,8737,3991,4113,4351,3346,763,5413,9538,2879,4870,8758,7509,7251,3028,9625,3825,7523,8398,6455,5988,4752,458,9425,1278,9866,4923,6949,2878,6039,6053,510,7375,9296,837,2649,2636,8712,8597,6554,6526,5546,4952,2013,9963,2442,6660,9431,779,5844,7862,5857,5382,7234,4068,8149,5367,1837,9707,9535,4779,1949,2123,5074,5269,149,1543,489,9301,5954,9395,4615,9624,2501,7762,853,9987,2752,4254,993,8755,3531,172,1070,1582,7982,4989,7704,9964,6708,5096,8054,2106,5848,7060,5066,9294,7645,2284,937,6483,6288,826,6288,7599,3966,2313,8883,260,3971,9303,4867,5625,9206,7937,4844,664,1612,3951,4099,8718,4390,1898,7026,8658,1711,4765,8701,1313,2826,1035,1422,2728,8241,3183,1019,272,7179,3630,423,8427,8723,6488,6960,243,8016,4956,9412,8791,7545,926,4828,1670,3836,1460,5933,5588,6465,1235,6293,7652,3428,7881,6213,4245,8187,4502,3823,4374,7518,8998,505,1915,4413,1584,9618,2293,5954,5143,5953,1881,6658,5326,1545,2549,2576,8598,6926,5087,7494,812,8371,4226,7372,808,8408,8608,8710,4907,2451,3193,7943,5566,9226,5046,25,8431,4864,6381,490,9193,9821,6703,6059,7864,9931,6568,2000,590,1577,7748,1638,14,9358,776,9500,1367,6951,2849,8163,9332,9570,6022,8480],
382801))
"""

#1578. Minimum Deletion Cost to Avoid Repeating Letters
"""
def minCost(s, cost):
    import itertools
    import operator
    import functools
    c=[list(map(lambda x:x[1],g)) for k,g in itertools.groupby(zip(list(s),cost),operator.itemgetter(0))]
    d=[x for x in c if len(x)>1]
    return functools.reduce(lambda x,y:x+y,map(lambda x: sum(x)-max(x),d)) if len(d) else 0
print(minCost("abc", [1,2,3]))
"""

#1391. Check if There is a Valid Path in a Grid
"""
def hasValidPath(grid):
    if grid[0][0]==4 or grid[-1][-1]==5:
        return False
    base=[[0 for x in range(len(grid[0]))] for y in grid[0]]
    def findpath(x,y):
        if x==0 and y==0:         
        if x==0:
"""
#1647. Minimum Deletions to Make Character Frequencies Unique
"""
def minDeletions(s):
    import collections
    base=collections.Counter(s)
    c=sorted(list(base.values()),reverse=True)
    cur_max=c[0]+1
    ans=0
    print(c)
    for x in c:
        if cur_max<=x:
            if cur_max==0:
                ans+=x
            else:
                ans+=x-cur_max+1
                cur_max-=1
        else:
            cur_max=x
    return ans 
print(minDeletions(s = "ageyjib"))
"""

#1648. Sell Diminishing-Valued Colored Balls
"""
def maxProfit(inventory, orders):
    base=sorted(inventory,reverse=True)+[0]
    ans=0
    width=1
    for x in range(len(base)):
        if (base[x]-base[x+1])*width>=orders:
            rows=orders//width
            ans+=rows*(base[x]+base[x]-rows+1)*width//2+(orders-rows*width)*(base[x]-rows)
            return ans%(10**9+7)
        else:
            orders-=width*(base[x]-base[x+1])
            ans+=width*(base[x]-base[x+1])*(base[x]+base[x+1]+1)//2
        width+=1
print(maxProfit(inventory = [3,5], orders = 6))
"""

#1488. Avoid Flood in The City
"""
def avoidFlood(rains):
    import collections
    import bisect
    p=collections.defaultdict(list)
    dry=[]
    ans=[]
    for x in range(len(rains)):
        if rains[x]>0:
            ans.append(-1)
            p[rains[x]].append(x)
            if len(p[rains[x]])==2:
                if not dry:
                    return []
                elif dry[-1]<p[rains[x]][0]:
                    return []
                else:
                    i=bisect.bisect_right(dry,p[rains[x]][0])
                    ans[dry.pop(i)]=rains[x]
                    p[rains[x]].pop(0)

        else:
            dry.append(x)
            ans.append(1)
    return ans
print(avoidFlood([1,0,2,3,0,1,2,0,3]))


import collections
    p=collections.defaultdict(int)
    dry=[]
    ans=[]
    for x in range(len(rains)):
        if rains[x]>0:
            ans.append(-1)
            p[rains[x]]+=1
            if p[rains[x]]>=2:
                if not dry:
                    return []
                else:
                    ans[dry.pop(0)]=rains[x]
        else:
            dry.append(x)
            ans.append(1)
    return ans
"""
#1492. The kth Factor of n
"""
def kthFactor(n, k):
    import math
    if n==1:
        return 1 if k==1 else -1
    elif n==2:
        return k if k<3 else -1
    else:
        c=[1]+[i for i in range(2,int(math.sqrt(n))+1) if n%i==0]
        if int(math.sqrt(n))**2==n:
            if k>=2*len(c):
                return -1
            elif k<=len(c):
                return c[k-1]
            else:
                return n//c[len(c)-k-1]
        else:
            if k>2*len(c):
                return -1
            elif k<=len(c):
                return c[k-1]
            else:
                 return n//c[len(c)-k]
print(kthFactor(4,4))
"""
#1498. Number of Subsequences That Satisfy the Given Sum Condition
"""
def numSubseq(nums, target):
    def cnt(n):
        return 2**n-1
    if len(nums)==1:
        return 1 if sum(nums)*2<=target else 0
    else:
        b=sorted(nums)
        res=0
        s=b[0]
        for x in b:
            if x+s<=target:
                res+=1
        return res
print(numSubseq(nums = [3,5,6,7], target = 9)) 
"""

#781. Rabbits in Forest
"""
def numRabbits(answers):
    import math
    import itertools
    answers.sort()
    c=[list(g) for k,g in itertools.groupby(answers)]
    return sum([math.ceil(len(x)/(x[0]+1))*(x[0]+1) for x in c])
print(numRabbits(answers = [1,1,2]))
"""

#1637. Widest Vertical Area Between Two Points Containing No Points
"""
def maxWidthOfVerticalArea(points):
    b=points.sort(key=lambda x: x[0])
    return max(map(lambda x:x[0][0]-x[1][0], zip(points[1:],points[:-1])))
print(maxWidthOfVerticalArea(points = [[3,1],[9,0],[1,0],[1,4],[5,3],[8,8]]))
"""

#791. Custom Sort String
"""
def customSortString(S, T):
    import collections
    s=collections.Counter(S)
    t=collections.Counter(T)
    ans=""
    for x in list(s.keys()):
        if x in t.keys():
            ans+=x*t[x]
    for y in list(t.keys()):
        if y not in s.keys():
            ans+=y*t[y]
    return ans

print(customSortString(S = "cba",T = "abcd"))
"""

#1471. The k Strongest Values in an Array
"""
def getStrongest(arr, k):
    arr.sort()
    m=arr[(len(arr)-1)//2]

    def conv(alist):
        ans=[]
        for x in alist:
            if x<0:
                ans.append(abs(x)-0.5)
            else:
                ans.append(x)
        return ans
    
    ans=list(zip(arr,conv([x-m for x in arr])))
    b=sorted(ans,key=lambda x:x[1],reverse=True)
    return [x[0] for x in b[:k]]
print(getStrongest(arr = [6,-3,7,2,11], k = 3))
"""

#1451. Rearrange Words in a Sentence
"""
def arrangeWords(text):
    import collections
    d=collections.defaultdict(list)
    ans=""
    for x in text.split(" "):
        d[len(x)].append(x)
    for y in sorted(d.keys()):
        ans+=" ".join(d[y])
        ans+=" "
    return ans[0].upper()+ans[1:].lower()

print(arrangeWords(text =  "To be or not to be"))
"""

#976. Largest Perimeter Triangle
"""
def largestPerimeter(A):
    A.sort(reverse=True)
    print(A)
    for i in range(len(A)-2):
        if A[i]<A[i+1]+A[i+2]:
            return A[i]+A[i+1]+A[i+2]
    return 0
print(largestPerimeter([3,6,2,3]))
"""

#1573. Number of Ways to Split a String
"""
def numWays(s):
    import collections
    if sum(map(int,list(s)))==0:
        return (len(s)-1)*(len(s)-2)//2
    elif sum(map(int,list(s)))%3!=0:
        return 0
    else:
        n=sum(map(int,list(s)))//3
        res=0
        for x in range(len(s)):
            if s[x]=="1":
                res+=1
                if res==n:
                    a=x
                if res==n+1:
                    b=x
                if res==2*n:
                    c=x
                if res==2*n+1:
                    d=x
    return (b-a)*(d-c)

print(numWays("0000"))
"""

#853. Car Fleet
"""
def carFleet(target, position, speed):
    time=[float((target-p)/s) for p,s in sorted(zip(position,speed))]
    tmp=0
    ans=0
    for t in time[::-1]:
        if t>tmp:
            ans+=1
            tmp=t
    return ans
print(carFleet(target = 12, position = [10,10], speed = [2,5]))
"""


#972.Equal Rational Number
"""
def isRationalEqual(self, S, T):
    def f(s):
        i = s.find('(')
        if i >= 0:
            s = s[:i] + s[i + 1:-1] * 20
        return float(s[:20])
    return f(S) == f(T)
"""

#818. Race Car
#Unsolved
"""
def racecar(target):
    import math
    import bisect
    base=[2**n-1 for n in range(15)]
    def getsteps(n):
        if n in base:
            return base.index(n)
        else:
            k=bisect.bisect_left(base,n)
            return k+1+min(getsteps(n-base[k-1]),getsteps(base[k]-n))
    return getsteps(target)
print(racecar(6))
"""

#41. First Missing Positive
"""
def firstMissingPositive(nums):
    nums.append(0)
    n=len(nums)
    print(n)
    for i in range(n):
        if nums[i]>=n or nums[i]<0:
            nums[i]=0
    for i in range(n):
        nums[nums[i]%n]+=n
    print(nums)
    for i in range(1,n):
        if nums[i]//n==0:
            return i
    return n
print(firstMissingPositive([]))
"""
#799. Champagne Tower
"""
def champagneTower(poured,query_row,query_glass):
    l=[poured]
    for i in range(query_row):
        l_new=[0]*(len(l)+1)
        for i in range(len(l)):
            pour=(l[i]-1)/2
            if pour>0:
                l_new[i]+=pour
                l_new[i+1]+=pour
        l=l_new
    return min(1,l[query_glass])
print(champagneTower(poured = 100000009, query_row = 33, query_glass = 17))
"""
"""
def maxLength(arr):
    s = [""]
    res = 0
    for i in arr:
        if len(set(i)) == len(i):
            for j in s[:]:
                if len(i) + len(j) == len(set(i + j)):
                    s.append(i + j)
                    if len(i + j) > res:
                        res = len(i + j)
    return res
print(maxLength(['abcde','abc','def']))"""

#1447. Simplified Fractions
"""
def simplifiedFractions(n):
    import math
    def gcd(p,q):
        while q!=0:
            p,q=q,p%q
        return p
    def is_co(x,y):
        return gcd(x,y)==1
    if n==1:
        return []
    else:
        ans=[]
        for x in range(2,n+1):
            for y in range(1,x):
                if is_co(x,y):
                    ans.append(str(y)+"/"+str(x))
        return ans
print(simplifiedFractions(6))
"""

#794. Valid Tic-Tac-Toe State
#unsolved


#658. Find K Closest Elements
"""
def findClosestElements(arr, k, x):
    l,r=0,len(arr)-k
    while l<r:
        mid=(l+r)//2
        if x-arr[mid]>arr[mid+k]-x:
            l=mid+1
        else:
            r=mid
    return arr[l:l+k]
print(findClosestElements(arr = [1,2,3,4,5], k = 4, x = -1))
"""
#424. Longest Repeating Character Replacement
"""
def characterReplacement(s, k):
    import collections
    if s=="":
        return 0
    i=0
    j=k+1
    start=s[i:j]
    base=collections.Counter(start)
    ans=k+1
    while j<len(s):
        if j-i-max(base.values())>k:
            base[s[i]]-=1
            i+=1
        else:
            ans=max(ans,j-i)
            base[s[j]]+=1
            j+=1           
    if j-i-max(base.values())>k:
        return ans
    else:
        return max(ans,j-i)
print(characterReplacement(s="DFADD",k=4))
"""

        

#1344. Angle Between Hands of a Clock
"""
def angleClock(hour, minutes):
    m=6*(minutes%60)
    h=30*(hour%12)+0.5*(minutes%60)
    return min( max(h,m)-min(m,h), min(m,h)+360-max(m,h)     )
print(angleClock(4,50))
"""

#1366. Rank Teams by Votes
"""
def rankTeams(votes):
    import collections
    import itertools
    p=list(votes[0])
    base=[''.join(a) for a in zip(*votes)]

    cnt=[collections.Counter(list(x)) for x in base]

    p.sort(key=lambda x: [-a[x] for a in cnt]+[x])
    print(cnt)

    return "".join(p)

print(rankTeams(votes = ["BCA","CAB","CBA","ABC","ACB","BAC"]))
"""

#1197. Minimum Knight Moves
#unsolved
"""
def minKnightMoves(o, p):
    x,y=max(abs(o),abs(p)),min(abs(o),abs(p))
    base={()}
"""


#1090. Largest Values From Labels
"""
def largestValsFromLabels(values, labels, num_wanted, use_limit):
    ans=0
    import collections
    d=collections.defaultdict(int)
    base=[[x,y] for x,y in zip(values,labels)]
    print(base)
    base.sort(key=lambda x : [-x[0],x[1]])
    print(base)
    
    n=len(base)
    l=0
    for x in range(n):
        if d[base[x][1]]<use_limit:
            ans+=base[x][0]
            d[base[x][1]]+=1
            l+=1
            if l==num_wanted:
                return ans
        else:
            continue
    return ans
print(largestValsFromLabels(
[3,2,3,2,1],
[1,0,2,2,1],
2,
1))
"""
#1134. Armstrong Number
"""
import functools
def isArmstrong(N):
    n=len(str(N))
    l=list(str(N))
    ans=[int(x)**n for x in l]
    return sum(ans)==N
print(isArmstrong(153))
"""

#159. Longest Substring with At Most Two Distinct Characters
"""
def lengthOfLongestSubstringTwoDistinct(s):
    import collections
    if len(s)<=2:
        return len(s)
    n=len(s)
    ans=2
    cut=-1
    d=collections.defaultdict(list)
    for x in range(n):
        print(d)
        if s[x] not in d.keys() and len(d.keys())==2:
            cut=min(d.values())
            ans=max(ans,x-cut)
            del d[s[cut]]
        ans=max(ans,x-cut)
        d[s[x]]=x
    return ans

print(lengthOfLongestSubstringTwoDistinct("ccaabbb"))
"""

#829. Consecutive Numbers Sum
"""
def consecutiveNumbersSum(N):
    m=1
    n=0
    k=N/m
    ans=0
    while k>=1:
        if k==int(k):
            ans+=1
        n+=m
        m+=1
        k=float((N-n)/m)
    return ans
print(consecutiveNumbersSum(15))
"""

#1499. Max Value of Equation
"""
def findMaxValueOfEquation(points, k):
    import heapq as hp
    q=[]
    ans=float('-inf')
    for x,y in points:
        while q and q[-1][0]<x-k:
            hp.heappop(q)
        if q:
            ans=max(ans,-q[0][0]+y+x)
        hp.heappush(q,(x-y,x))
    return ans
"""

#1262. Greatest Sum Divisible by Three
"""
def maxSumDivThree(nums):
    o1, o2 = float('inf'), float('inf')
    e1, e2 = float('inf'), float('inf')
    for x in nums:
        if x%3==1:
            if x <= o1:
                o1, o2 = x, o1
            elif x < o2:
                o2 = x
        elif x%3==2:
            if x <= e1:
                e1, e2 = x, e1
            elif x < e2:
                e2 = x
        else:
            continue
    ans=sum(nums)
    if ans%3==0:
        return ans
    elif ans%3==1:
        return ans-min(e1+e2,o1)
    else:
        return ans-min(o1+o2,e1) 

print(maxSumDivThree( nums = [2,3,36,8,32,38,3,30,13,40]))
"""


#1023. Camelcase Matching
"""
return [re.match("^[a-z]*" + "[a-z]*".join(p) + "[a-z]*$", q) != None for q in qs]
def camelMatch(queries, pattern):
    ans=[]
    for w in queries:
        tmp=""
        for c in w:
            if ord(c)<=90:
                tmp+=c
        ans.append(tmp==pattern)
    return ans
print(camelMatch(queries = ["FooBar","FooBarTest","FootBall","FrameBuffer","ForceFeedBack"], pattern = "FB"))
"""


#356. Line Reflection
"""
def isReflected(points):
    import collections
    d=collections.defaultdict(list)
    for x,y in points:
        d[y].append(x)
    ans=set()
    def gety(a):
        nonlocal ans
        if len(a)==1:
            ans.add(a[0])
        else:
            ans|=set(map(lambda x:(x[0]+x[1])/2,zip(a,a[::-1])))
    for x in d.keys():
        gety(d[x])
    return len(ans)==1

print(isReflected([[1,2],[2,2],[3,2],[4,2]]))
"""

#1230. Toss Strange Coins
#unsolved

#1272. Remove Interval
"""
def removeInterval(intervals, toBeRemoved):
    ans=[]
    for x in intervals:
        if x[1]<=toBeRemoved[0] or x[0]>=toBeRemoved[1]:
            ans.append(x)
        elif x[0]<=toBeRemoved[0] and x[1]>=toBeRemoved[1]:
            ans.append([x[0],toBeRemoved[0]])
            ans.append([x[1],toBeRemoved[1]])
        elif x[0]>=toBeRemoved[0] and x[1]<=toBeRemoved[1]:
            continue
        elif x[0]<=toBeRemoved[0] and x[1]>=toBeRemoved[0]:
            ans.append([x[0],toBeRemoved[0]])
        else:
            ans.append([toBeRemoved[1],x[1]])
    return ans
print(removeInterval(intervals = [[-5,-4],[-3,-2],[1,2],[3,5],[8,9]], toBeRemoved = [-1,4]))
"""

#805. Split Array With Same Average
#unsolved


#84. Largest Rectangle in Histogram
"""
def largestRectangleArea(heights):
    n=len(heights)
    if n==1:
        return sum(heights)   
    l=[1]*n
    r=[1]*n
    for i in range(1,n):
        j=i-1
        while j and heights[j]<=heights[i]:
            j-=l[j]
        l[i]=i-j
    for i in range(n-2,-1,-1):
        j=i+1
        while j<n and heights[i]<=heights[j]:
            j+=r[j]
        r[i]=j-i

    return max(x[0]*(x[1]+x[2]-1) for x in zip(heights,l,r))
print(largestRectangleArea([2,1,5,6,2,3]))
"""

#76. Minimum Window Substring
"""
def minWindow(s, t):
    import collections
    d=collections.Counter(t)
    base=collections.Counter(s)
    for x in d.keys():
        if base[x]<d[x]:
            return ""
    k=d.keys()
    i,j=0,0
    n=len(s)
    r=[0,0]
    ans=n
    key=[]
    res=[n-1,0
    ]
    while j<n:
        if s[j] in k:
            d[s[j]]-=1
            key.append([s[j],j])
            print(key)
            if max(d.values())<=0:
                print("istime")
                while d[key[0][0]]<0:
                    tmp=key.pop(0)
                    d[tmp[0]]+=1
                    i=tmp[1]+1
                if key[-1][1]-key[0][1]+1<ans:
                    ans=key[-1][1]-key[0][1]+1
                    res=[key[-1][1],key[0][1]]
        j+=1
        print(res)
    return s[res[1]:res[0]+1]
print(minWindow(s = "aa", t = "aa")) 
"""

#1177. Can Make Palindrome from Substring
"""
def canMakePaliQueries(s, queries):
    ans=[]
    for x in queries:
        tmp=s[x[0]:x[1]+1]
        n=len(tmp)
        dif=0
        for i in range(n//2):
            if tmp[i]!=tmp[::-1][i]:
                dif+=1
        if dif<=x[2]:
            ans.append('true')
        else:
            ans.append('false')
    return ans
print(canMakePaliQueries(s = "abcda", queries = [[3,3,0],[1,2,0],[0,3,1],[0,3,2],[0,4,1]]))
"""
#1209. Remove All Adjacent Duplicates in String II
"""
def removeDuplicates(s, k):
    st=[]
    ans=""
    for i in s:
        if not st or st[-1][0]!=i:
            st.append([i,1])
        else:
            if st[-1][1]==k-1:
                st.pop()
            else:
                st[-1][1]+=1
    for x in st:
        ans+=x[0]*x[1]
    return ans
print(removeDuplicates(s = "deeedbbcccbdaa", k = 3))
"""

#179. Largest Number
"""
def largestNumber(nums):
    import functools
    num = [str(x) for x in nums]
    num.sort(key = functools.cmp_to_key(lambda b, a: ((a+b)>(b+a))-((a+b)<(b+a)) ))
    print(num)

    return ''.join(num).lstrip('0') or '0'
print(largestNumber([3,30,34,5,9]))
"""
#1353. Maximum Number of Events That Can Be Attended

#1057. Campus Bikes
"""
def assignBikes(W, B):
    ans=[-1]*len(W)
    used=set()
    for d,w,b in sorted([abs(W[i][0]-B[j][0])+abs(W[i][1]-B[j][1]),i,j] for i in range(len(W)) for j in range(len(B))):
        if ans[w]==-1 and b not in used:
            ans[w]=b
            used.add(b)
    return ans
print(assignBikes([[0,0],[2,1]], [[1,2],[3,3]]))
"""

#1552. Magnetic Force Between Two Balls
"""
def maxDistance(p, m):
    p.sort()
    def getb(d):
        if len(p)==1:
            return 1
        cur=p[0]
        ans=1
        for x in p[1:]:
            if x-cur>=d:
                ans+=1
                cur=x
        return ans
    i=1
    j=p[-1]-p[0]
    while i<j:
        mid=j-(j-i)//2
        print(i,j,mid)
        if getb(mid)>=m:
            i=mid
        else:
            j=mid-1
    return i
print(maxDistance(p = [1,2,3,4,5,10000], m = 2))
"""       

#1029. Two City Scheduling
"""
def twoCitySchedCost(costs):
    ans=0
    toa=[]
    tob=[]
    eq=0
    a,b=0,0
    n=len(costs)
    for x in costs:
        if x[0]==x[1]:
            eq+=1
            ans+=x[0]
        elif x[0]>x[1]:
            tob.append(x)
            b+=1
        else:
            toa.append(x)
            a+=1
    if max(a,b)<=n//2:
        ans=ans+sum(x[0] for x in toa)+sum(x[1] for x in tob)
    else:
        if a>b:
            m=a-n//2
            mlist=sorted(toa,key=lambda x: x[1]-x[0])[:m]
            ans=ans+sum(x[1] for x in tob)+sum(x[1] for x in mlist)+sum(x[0] for x in sorted(toa,key=lambda x: x[1]-x[0])[m:])
        else:
            m=b-n//2
            mlist=sorted(tob,key=lambda x: x[0]-x[1])[:m]
            ans=ans+sum(x[0] for x in toa)+sum(x[0] for x in mlist)+sum(x[1] for x in sorted(tob,key=lambda x: x[0]-x[1])[m:])
    return ans
print(twoCitySchedCost(costs = [[259,770],[448,54],[926,667],[184,139],[840,118],[577,469]]))
"""
#866. Prime Palindrome
"""
def primePalindrome(N):
    import math
    x=N+1
    nofind=True
    while nofind:
        if [i for i in range(2,int(math.sqrt(x))+1) if x%i==0]==[] and str(x)==str(x)[::-1]:
            break
        x+=1 
    return x
print(primePalindrome(6))
"""
#875. Koko Eating Bananas
"""
def minEatingSpeed(piles, H):
    import math
    def geth(p,s):
        return sum(math.ceil(x/s) for x in p)    
    i=1
    j=max(piles)
    while i<j:
        mid=i+(j-i)//2
        if geth(piles,mid)<=H:
            j=mid
        else:
            i=mid+1
    return i
print(minEatingSpeed(piles = [30,11,23,4,20], H = 5))
"""

#1167. Minimum Cost to Connect Sticks
"""
def connectSticks(sticks):
    import heapq
    heapq.heapify(sticks)
    ans=0
    while len(sticks)>1:
        x,y=heapq.heappop(sticks),heapq.heappop(sticks)
        ans+=x+y
        heapq.heappush(sticks,x+y)
    return ans
print(connectSticks(sticks = [1,8,3,5]))

"""

#233. Number of Digit One
"""
def countDigitOne(n):
    s="0"+str(n)
    ans=0
    for i in range(1,len(s)):
        base=int(s[:i])*10**(len(s)-i-1)
        print(s[:i],int(s[:i]),base)
        if s[i]=="1":
            if i<len(s)-1:
                base+=int(s[i+1:])+1
        if s[i]!="1" and s[i]!="0":
            base+=10**(len(s)-i-1)
        print(base)
        ans+=base
    return ans
print(countDigitOne(100))
"""

#315. Count of Smaller Numbers After Self
"""
def countSmaller(nums):
    import sortedcontainers
    cnt=[]
    sortnum=sortedcontainers.SortedList(nums)
    for x in nums:
        i=sortnum.index(x)
        cnt.append(i)
        sortnum.remove(x)
    return cnt
print(countSmaller(nums = [5,2,6,1]))
"""

#659. Split Array into Consecutive Subsequences


#681. Next Closest Time
"""
def nextClosestTime(time):
    def addsec(t):
        res_s=(int(t[3:])+1)%60
        to_h=(int(t[3:])+1)//60
        res_h=int(t[:2])+to_h
        if res_h==24:
            return "00:00"
        else:
            return ("0"+str(res_h))[-2:]+":"+("0"+str(res_s))[-2:]
    aset=set([int(time[0]),int(time[1]),int(time[3]),int(time[4])])

    t=addsec(time)
    m=0
    while m<2400:
        if not (set([int(t[0]),int(t[1]),int(t[3]),int(t[4])]).union(aset) <=aset):
            t=addsec(t)
        else:
            break
    return t
print(nextClosestTime("22:22"))
"""

#667. Beautiful Arrangement II

#247. Strobogrammatic Number II

#412. Fizz Buzz
"""
def fizzBuzz(n):
    ans=[]
    for x in range(1,n+1):
        if x%3==0 and x%5==0:
            ans.append("FizzBuzz")
        elif x%3==0:
            ans.append("Fizz")
        elif x%5==0:
            ans.append("Buzz")
        else:
            ans.append(str(x))
    return ans
print(fizzBuzz(15))
"""

#436. Find Right Interval
"""
def findRightInterval(intervals):
    import bisect
    l = sorted((e[0], i) for i, e in enumerate(intervals))
    res = []
    for e in intervals:
        r = bisect.bisect_left(l, (e[1],))
        res.append(l[r][1] if r < len(l) else -1)
    return res
print(findRightInterval([[1,2],[3,2],[2,4],[5,6],[4,5]]))
"""

#487. Max Consecutive Ones II
"""
def findMaxConsecutiveOnes(nums):
    import itertools
    a=[(i,len(list(k))) for i,k in itertools.groupby(nums)]
    ans=0
    n=len(a)
    print(a)
    if n==1:
        return n if a[0][0]==1 else 1
    

    for x in range(n):
        if a[x][0]==1:
            ans=max(ans,a[x][1]+1)
        if x<n-2 and a[x+1][1]==1:
            ans=max(ans,a[x][1]+a[x+2][1]+1)
    return ans
print(findMaxConsecutiveOnes([1,0,1,1,0]))

"""

#484. Find Permutation


#713. Subarray Product Less Than K
"""
def numSubarrayProductLessThanK(nums,k):
    if not nums:
        return 0
    j=0
    res=1
    cnt=0
    for i in range(len(nums)):
        res*=nums[i]
        if res>=k:
            while j<=i and res>=k:
                res/=nums[j]
                j+=1
        cnt+=i-j+1
    return cnt
print(numSubarrayProductLessThanK([10,2,8,6,3],13))
"""

#660. Remove 9
"""
def newInteger(n):
    ans=""
    while n:
        ans+=str(n%9)
        n//=9
    return ans
print(newInteger(100))
"""


#1012. Numbers With Repeated Digits
"""
def numDupDigitsAtMostN(N):
    def A(m, n):
        res = 1
        for i in range(n):
            res *= m
            m -= 1
        return res
    l=list(map(int,str(N)))
    ans=0
    for x in range(1,len(l)):
        ans+=9*A(9,x-1)
    seen=set()
    for x in range(len(l)):
        for y in range(0 if x!=0 else 1, l[x]):
            if y not in seen:
                ans+=A(9-x,n-x-1)
        if x in seen:
            break
        seen.add(x)
    return N-ans
print(numDupDigitsAtMostN(1000))
"""


#1118. Number of Days in a Month
"""
def numberOfDays(Y, M):
    m=(1,3,5,7,8,10,12)
    l=(4,6,9,11)
    if M in m:
        return 31
    elif M in l:
        return 30
    else:
        if Y%4!=0 or (Y%100==0 and Y%400!=0):
            return 28
        else:
            return 29
print(numberOfDays(Y = 1900, M = 2))
"""

#910. Smallest Range II
"""
def smallestRangeII(A, K):
    A.sort()
    d=A[-1]-A[0]
    if d<=K:
        return d
    else:
        n=len(A)
        for j in range(n-1):
            cur=[A[0]+2*K,A[j]+2*K,A[j+1],A[-1]]
            d=min(d,max(cur)-min(cur))
        return d
print(smallestRangeII(A = [1,3,6], K = 3))
"""

#276. Paint Fence
"""
def numWays(n, k):
    import math
    if n==1:
        return k
    if k==1:
        return 1 if n==1 else 0
    ans=0
    for x in range(math.ceil(n/2),n):
        ans+=k*(k-1)**(x-1)*math.comb(x,n-x)
    ans+=k*(k-1)**(n-1)
    return ans
print(numWays(10,2))
"""

#1283. Find the Smallest Divisor Given a Threshold
"""
def smallestDivisor(nums, threshold):
    import math
    i=1
    j=max(nums)
    while i<j:
        print(i,j)
        m=i+(j-i)//2
        res=sum(map(lambda x:math.ceil(x/m),nums))
        if res>threshold:
            i=m+1
        else:
            j=m
    return i
print(smallestDivisor(nums = [1,2,5,9], threshold = 6))
"""

#1326. Minimum Number of Taps to Open to Water a Garden
"""
def minTaps(n, ra):
    st=[]
    for p in range(len(ra)):
        if not ra[p]:
            continue
        else:
            l,r=p-ra[p],p+ra[p]
            if not st:
                pass
            else:
                if r<=st[-1][1]:
                    continue
                elif l>=st[-1][0] and st[-1][1]>=n:
                    continue
                else:
                    if l<=0:
                        st=[]
                    else:
                        while st and l<=st[-1][0]:
                            st.pop()
                        while len(st)>=2 and l<=st[-2][1]:
                            st.pop()
            st.append([l,r])
    if not st:
        return -1
    if len(st)==1:
        if st[0][0]>0 or st[0][1]<n:
            return -1
        else:
            return 1
    else:
        if not any(x[0]<=0 for x in st) or not any(x[1]>=n for x in st):
            return -1
        else:
            for z in range(1,len(st)):
                if st[z][0]>st[z-1][1]:
                    return -1
            return len(st)

print(minTaps(
10,
[0,2,0,0,0,0,0,2,0,0,1]))
"""


#869. Reordered Power of 2
def reorderedPowerOf2(N):
    import collections
    c=collections.Counter(str(N))
    n=1
    while len(str(n))<=len(str(N)):
        if collections.Counter(str(n))==c:
            return True
        n*=2
    return False
print(reorderedPowerOf2(1214124))
