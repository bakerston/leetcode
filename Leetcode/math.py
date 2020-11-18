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
def maxCoins(piles):
    """
    alen=int(len(piles)/3)
    base=sorted(piles)[::-1]
    ans=[base[1+2*x] for x in range(alen)]
    return sum(ans)"""
    return sum(sorted(piles)[len(piles) // 3::2])
print(maxCoins(piles = [2,4,1,2,7,8]))