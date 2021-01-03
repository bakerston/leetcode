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
"""
def concatenatedBinary(n):
    ans=1
    for x in range(2,n+1):
        ans=ans*2**(len(bin(x)[2:]))+x
        ans%=10**9+7
    return ans
print(concatenatedBinary(3121))
"""

#5619. Minimum Incompatibility
"""
def minimumIncompatibility(nums, k):
    import itertools

"""

#5609. Count the Number of Consistent Strings
"""
def countConsistentStrings(allowed, words):
    base=set(allowed)
    return len([x for x in words if set(x)|base<=base])
print(countConsistentStrings(allowed = "cad", words = ["cc","acd","b","ba","bac","bad","ac","d"]))
"""

#5610. Sum of Absolute Differences in a Sorted Array
"""
def getSumAbsoluteDifferences(nums):
    if len(nums)==2:
        return [abs(nums[0]-nums[1])]*2
    else:
        f=nums[0]
        n=len(nums)
        base=sum(abs(f-x) for x in nums[1:])
        ans=[base]
        for i in range(1,n):
            base-=(n-2*i)*(nums[i]-nums[i-1])
            ans.append(base)
        return ans
print(getSumAbsoluteDifferences([1,4,6,8,10]))
"""

#5612. Delivering Boxes from Storage to Ports
"""
def boxDelivering(boxes, portsCount, maxBoxes, maxWeight):
    import itertools
    n=len(boxes)
    dp=[float('inf')]*(n+1)
    dp[0]=0
    for i in range(1,n+1):
        max_w=maxWeight
        max_b=maxBoxes
        j=i-1
        portset=[]
        cur_port=0
        while j>=0 and max_w>=boxes[j][1] and max_b>=1:
            max_w-=boxes[j][1]
            max_b-=1
            if not portset or boxes[j][0]!=portset[-1]:
                cur_port+=1
            portset.append(boxes[j][0])
            dp[i]=min(dp[i],dp[j]+cur_port+1)
            j-=1
    print(dp)
    return dp[n]

print(boxDelivering( boxes = [[2,4],[2,5],[3,1],[3,2],[3,7],[3,1],[4,4],[1,3],[5,2]], portsCount = 5, maxBoxes = 5, maxWeight = 7))
"""

#5625. Count of Matches in Tournament
"""
def numberOfMatches(n):
    ans=0
    while n!=1:
        ans+=n//2
        if n%2==0:
            n/=2
        else:
            n=n//2+1
        
    return int(ans)
print(numberOfMatches(14))
"""

#5626. Partitioning Into Minimum Number Of Deci-Binary Numbers
"""
def minPartitions(n):
    return max(map(int,list(n)))
print(minPartitions(n = "27346209830709182346"))
"""

#5627. Stone Game VII
"""
def stoneGameVII(stones):
    presum=[0]+stones[:]
    for i in range(1,len(presum)):
        presum[i]+=presum[i-1]
    def getsc(i,j):
        return presum[j+1]-presum[i]
    n=len(stones)
    dp=[[0]*n for _ in range(n)]
    
    #for i in range(n-1,-1,-1):
    #    for j in range(i+1,n):
    #        dp[i][j]=max(getsc(i+1,j)-dp[i+1][j],getsc(i,j-1)-dp[i][j-1])
    for x in range(1,n):
        for y in range(x-1,-1,-1):
            dp[y][x]=max(getsc(y+1,x)-dp[y+1][x], getsc(y,x-1)-dp[y][x-1])
    
    
    return dp[0][n-1]
            

print(stoneGameVII([5,3,1,4,2]))
"""

#5245. Maximum Height by Stacking Cuboids
"""
def maxHeight(cuboids):
    dic={}
    cuboids.sort(key=lambda x:x[2])
    for x in cuboids:
        for key in dic.keys():
            if max(x[:2])<=min(key):
                dic(tuple(x[:2]))=dic[key]+x[2]
"""     

#1694. Reformat Phone Number
"""
def reformatNumber(number):
    ans=""
    for x in number:
        if ord(x) in range(48,58):
            ans+=x
    res=""
    if len(ans)%3==0:
        for x in range(len(ans)):
            if x%3==0 and x!=0:
                res+="-"
            res+=ans[x]
    elif len(ans)%3==1:
        for x in range(len(ans)):
            if x%3==0 and x!=0 and x!=len(ans)-1:
                res+="-"
            if x==len(ans)-2:
                res+="-"
            res+=ans[x]
    else:
        for x in range(len(ans)):
            if x%3==0 and x!=0:
                res+="-"
            res+=ans[x]

    return res
print(reformatNumber("12345612"))
"""

#1695. Maximum Erasure Value
"""
def maximumUniqueSubarray(nums):
    if len(nums)==1:
        return nums[0]
    ans=0
    cur=nums[0]
    seen=set()
    seen.add(cur)
    i=0
    j=1
    n=len(nums)
    while j<n:
        if nums[j] not in seen:
            cur+=nums[j]
            ans=max(ans,cur)
            seen.add(nums[j])
            j+=1
        else:
            while nums[i]!=nums[j]:
                cur-=nums[i]
                seen.remove(nums[i])
                i+=1
            cur-=nums[i]
            seen.remove(nums[i])
            i+=1
    return ans
print(maximumUniqueSubarray(nums = [5,2,1,2,5,2,1,2,5]))
"""

#1696. Jump Game VI
"""
def maxResult(nums, k):
    import heapq
    hp=[]
    for i in reversed(range(len(nums))):
        while hp and hp[0][1]-i>k:
            heapq.heappop(hp)
        ans=nums[i]-hp[0][0] if hp else nums[i]
        heapq.heappush(hp,(-ans,i))
    return ans
print(maxResult(nums = [1,-1,-2,4,-7,3], k = 2))
"""

#5621. Number of Students Unable to Eat Lunch

#def cnts(students, sandwiches):
#    import collections
#    c=collections.Counter(students)
#    ans=0
#    for x in sandwiches:
#        if c[x]>0:
#            c[x]-=1
#            ans+=1
#        else:
#            break
#    return len(students)-ans
#print(cnts(students = [1,1,1,0,0,1], sandwiches = [1,0,0,0,1,1]))


#5622. Average Waiting Time
"""
def averageWaitingTime(customers):
    n=len(customers)
    if n==1:
        return customers[0][1]
    cur=customers[0][0]
    ans=0
    for s,t in customers:
        if s>=cur:
            ans+=t
            cur=s+t
        else:
            ans+=cur-s+t
            cur+=t
    return ans/n
print(averageWaitingTime(customers = [[5,2],[5,4],[10,3],[20,1]]))
"""

#5623. Maximum Binary String After Change
"""
def maximumBinaryString(binary):
    import itertools
    import collections
    c=collections.Counter(binary)
    if '0' in c.keys():
        if c['0']==1:
            return binary
        else:
            numz=c['0']
            numo=c['1']
            i=binary.index('0')
            return '1'*(i+numz-1)+'0'+'1'*(len(binary)-i-numz)
    else:
        return binary
print(maximumBinaryString("000110"))
"""

#5637. Determine if String Halves Are Alike]
"""
def halvesAreAlike(s):
    d=set(['a','e','i','o','u','A','E','I','O','U'])
    def getvo(s):
        ans=0
        for x in s:
            if x in d:
                ans+=1
        return ans
    n=len(s)
    return getvo(s[:n//2])==getvo(s[n//2:])
print(halvesAreAlike(s = "AbCdEfGh"))
"""

#5638. Maximum Number of Eaten Apples



#5210. Where Will the Ball Fall
"""
def findBall(grid):
    ans=[]
    n=len(grid)
    m=len(grid[0])

    for x in range(m):
        cur=x
        y=0
        while cur>=0 and y<n:
            if grid[y][cur]==1:
                if cur==m-1 or grid[y][cur+1]==-1:
                    cur=-1
                    break
                else:
                    cur+=1
            if grid[y][cur]==-1:
                if cur==0 or grid[y][cur-1]==1:
                    cur=-1
                    break
                else:
                    cur-=1
            y+=1
        ans.append(cur)
        print(x,ans)
    return ans
print(findBall([[-1]]))
""" 

#5641. Maximum Units on a Truck
"""
def maximumUnits(boxTypes, truckSize):
    boxTypes.sort(key=lambda x:x[1],reverse=True)
    ans=0
    n=truckSize
    for b,v in boxTypes:
        if  n>b:
            ans+=b*v
            n-=b
        else:
            ans+=n*v
            return ans
    return ans
print(maximumUnits([[5,10],[2,5],[4,7],[3,9]],
10))
"""

#5642. Count Good Meals
"""
def countPairs(deliciousness):
    import collections
    c=collections.Counter(deliciousness)
    ans=0
    for x in c.keys():
        for index in range(22):
            if 2**index-x in c and 2**index-x>=x:
                if 2**index-x==x:
                    ans+=(c[x]*(c[x]-1))//2
                else:
                    ans+=c[x]*c[2**index-x]
    print(c)
    return ans%(10**9+7)
print(countPairs([149,107,1,63,0,1,6867,1325,5611,2581,39,89,46,18,12,20,22,234]))
"""

#5643. Ways to Split Array Into Three Subarrays
"""
def waysToSplit(nums):
    import bisect
    import math
    l=[0]*(len(nums))
    for i in range(1,len(nums)):
        l[i]=l[i-1]+nums[i-1]
    l=l[1:]
    s=sum(nums)
    print(l)
    for i in range(len(l)-1,-1,-1):
        if l[i]<=2*s/3 and i>0:
            ans=0
            print(i)
            while i>0:
                cur=l[i]
                hf=cur//2
                j=bisect.bisect_right(l,hf)
                k=bisect.bisect_left(l,hf)
                print(i,j,k)
                if j>0:
                    if k>0:
                        ans+=j-k+1
                    else:
                        ans+=j-k
                i-=1

            return ans              
    return -1
print(waysToSplit([1,2,2,2,5,0]))

"""
#5644. Minimum Operations to Make a Subsequence

def minOperations(target, arr):
    import collections
    import bisect
    d={x:i for i,x in enumerate(target)}
    res=[]
    for x in arr:
        if x in d:
            res.append(d[x])
       
    s = []
    for r in res:
        pos = bisect.bisect_left(s, r)
        if pos == len(s):
            s.append(r)
        else:
            s[pos] = r
    return len(target)-len(s)
print(minOperations(target = [6,4,8,1,3,2], arr = [4,7,6,2,3,8,6,1]))

                    