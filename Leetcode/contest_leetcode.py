#20201121
#1
"""
def arrayStringsAreEqual( word1, word2):
    import functools
    return functools.reduce(lambda x,y:x+y, word1)==functools.reduce(lambda x,y:x+y, word2)
print(arrayStringsAreEqual( word1  = ["abc", "d", "defg"], word2 = ["abcddefg"]))
"""
#2
"""
def getSmallestString(n, k):
    z=(k-n)//25
    res=(k-n)%25
    a=n-z-1 if res!=0 else n-z
    print(a,res,z)
    if res==0:
        return "a"*a+"z"*z
    else:
        return "a"*a+chr(ord("a")+res)+"z"*z
print(getSmallestString(9,36))
"""

#3.
"""
def waysToMakeFair(nums):
    d={}
    osum=sum([nums[x] for x in range(len(nums)) if x%2==1])
    esum=sum([nums[x] for x in range(len(nums)) if x%2==0])
    preo=0
    pree=0
    print(osum,esum)
    for x in range(len(nums)):
        if x%2==0:
            post_e=esum-pree-nums[x]
            post_o=osum-preo
        else:
            post_e=esum-pree
            post_o=osum-preo-nums[x]
        d[x]=[preo,pree,post_o,post_e]
        if x%2==0:
            pree+=nums[x]
        else:
            preo+=nums[x]
    ans=0
    for x in range(len(nums)):
        if d[x][0]+d[x][3]==d[x][1]+d[x][2]:
            ans+=1
    return ans
print(waysToMakeFair([1,2,3]))
"""

#4
"""
def minimumEffort(tasks):
    def merg(t1,t2):
        return t1[0]+t2[0],min(max(t1[1]+t2[0],t2[1]),max(t1[0]+t2[1],t1[1]))
    base=sorted(tasks,key=lambda x:x[1]-x[0])[::-1]
    return functools.reduce(merg,base)[1]

print(minimumEffort(tasks = [[1,3],[2,4],[10,11],[10,12],[8,9]]))
"""

#biweek
#5557.Maximum Repeating Substring
"""
def maxRepeating(sequence, word):
    import re
    s=len(sequence)
    n=len(word)
    if s<n:
        return 0
    elif s==n:
        return 1 if sequence==word else 0
    else:
        ans=0
        for x in range(1,s//n+1):
            if re.search(word*x,sequence):
                ans=max(ans,x)
        return ans 
print(maxRepeating(sequence = "abababab", word = "b")) 
"""

#5558. Merge In Between Linked Lists
#Definition for singly-linked list.
"""
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeInBetween(self, list1: ListNode, a, b, list2):
"""

#Weekly Contest 217
#5613. Richest Customer Wealth
"""
def maximumWealth( accounts):
    return max(map(sum,accounts))
print(maximumWealth(accounts = [[2,8,7],[7,1,3],[1,9,5]]))
"""

#5614. Find the Most Competitive Subsequence
"""
def mostCompetitive(nums, k):
    s=[]
    for i,a in enumerate(nums):
        while s and s[-1]>a and len(s)+len(nums)-i-1>=k:
            s.pop()
        if len(s)<k:
            s.append(a)
    return s
    
print(mostCompetitive(nums = [2,4,3,3,5,4,9,6], k = 4))

"""



#5615. Minimum Moves to Make Array Complementary
"""
def minMoves(nums, limit):
    import collections
    n=len(nums)
    a=nums[:n//2]
    b=nums[-1:-(n//2+1):-1]
    base=[x+y for x,y in zip(a,b)]
    ans=collections.defaultdict(list)
    for x in range(len(a)):
        ans[a[x]+b[x]].append([min(a[x],b[x])+1,max(a[x],b[x])+limit])

    return ans

print(minMoves([1,2,5,4,2,3,1,2],5))
"""

#5616. Minimize Deviation in Array
"""
def minimumDeviation(nums):
"""

#5617. Goal Parser Interpretation
"""
def interpret(command):
    ans=""
    n=len(command)
    x=0
    while x<n:
        if command[x]=="(":
            if command[x+1]=="a":
                ans+="al"
                x+=4
            else:
                ans+="o"
                x+=2
        else:
            ans+="G"
            x+=1
    return ans
print(interpret(command = "G()()()()(al)"))
"""

#5618. Max Number of K-Sum Pairs
"""
def maxOperations(nums, k):
    import collections
    c=collections.Counter(nums)
    s=set(c.keys())
    l=sorted(list(c.keys()))
    ans1=0
    ans2=0
    for x in l:
        if k-x in s:
            if k-x==x:
                ans1+=c[x]//2
            else:
                ans2+=min(c[x],c[k-x])
    return ans1+ans2//2
print(maxOperations(nums = [3,1,3,4,3], k = 6))
"""

#5620. Concatenation of Consecutive Binary Numbers
def concatenatedBinary(n):
    ans=1
    for x in range(2,n+1):
        ans=ans*2**(len(bin(x)[2:]))+x
        ans%=10**9+7
    return ans
print(concatenatedBinary(3121))
