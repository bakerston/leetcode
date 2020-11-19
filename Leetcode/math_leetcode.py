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
def minCost(s, cost):
    import itertools
    import operator
    import functools
    c=[list(map(lambda x:x[1],g)) for k,g in itertools.groupby(zip(list(s),cost),operator.itemgetter(0))]
    d=[x for x in c if len(x)>1]
    return functools.reduce(lambda x,y:x+y,map(lambda x: sum(x)-max(x),d)) if len(d) else 0
print(minCost("abc", [1,2,3]))

