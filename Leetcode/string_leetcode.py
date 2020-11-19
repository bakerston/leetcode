#1528 Shuffle string
"""def restoreString(s,indices):  
    res=list(s)
    for x in range(len(indices)):
        res[indices[x]]=s[x]
    return ''.join(res)
print(restoreString("codeleet",[4,5,6,7,0,2,1,3]))
"""

#1021. Remove Outermost Parentheses
"""def removeOuterParentheses(S):
    tmp=[]
    ans=""
    index=0
    for i in list(S):
        if i=="(":
            index+=1
            if index>1:
                ans+=i
        else:
            index-=1
            if index>0:
                ans+=i
    return ans

print(removeOuterParentheses("(()())(())(()(()))"))"""
#1614. Maximum Nesting Depth of the Parentheses
"""def maxDepth(s):
    ans=0
    maxnum=0
    for i in list(s):
        if i=="(":
            ans+=1
        elif i==")":
            ans-=1
        if ans>maxnum:
            maxnum=ans
    return maxnum
print(maxDepth("1+(2*3)/(2-1)"))
"""
#1221. Split a String in Balanced Strings
"""
def balancedStringSplit(s):
    mark,tmp,ans=s[0],1,0
    for i in range(1,len(s)):
        if tmp>=2:
            if s[i]==mark:
                tmp+=1
            else:tmp-=1
        elif tmp==1:
            if s[i]==mark:
                tmp+=1
            else:
                tmp=0
                ans+=1
        else:
            mark=s[i]
            tmp+=1
    return ans
print(balancedStringSplit("RLLLLRRRLR"))
def getWinner(arr, k):
    defender = arr[0]
    tmp = 0
    for i in range(1, len(arr)):
        if arr[i] > defender:
            defender = arr[i]
            tmp = 0
        tmp += 1
        if (tmp == k): break
    return defender
print(getWinner([1,2,3],1))"""

#1556. Thousand Separator
"""
def thousandSeparator(n):
    base=list(str(n))
    key=(len(base)-1)//3
    print
    if key ==0:
        return str(n) 
    else:
        for x in range(key,0,-1):
            base.insert(-3*x,".")
        return "".join(base)
print(thousandSeparator(1236))"""

#1557. Minimum Number of Vertices to Reach All Nodes
"""
def findSmallestSetOfVertices(n, edges):
    base=[x for x in range(n)]
    fo=list(set([x[0] for x in edges]))
    to=list(set([x[1] for x in edges]))
    only_from=list(set(fo)-set(to))
    link=set(fo+to)
    iso=list(set(base)-link)
    print(iso)
    return iso+only_from
print(findSmallestSetOfVertices(  n = 5, edges = [[0,1],[2,1],[3,1],[1,4],[2,4]]))
"""
"""
    adic={x:[x] for x in range(n)}
    for x in edges:
        adic[x[0]]=list(set().union(adic[x[0]],x[1]))
        if x[1] in adic.keys():
            """


#1653. Minimum Deletions to Make String Balanced
"""
def minimumDeletions(s):
    n=len(s)
    dp=[0]*n
    bnum=0
    if s[0]=="b":
        bnum=1
    for i in range(1,n):
        if s[i]=="a":
            dp[i]=min(dp[i-1]+1,bnum)
        else:
            dp[i]=dp[i-1]
            bnum+=1
    return dp[-1]

print(minimumDeletions( "aaabaaaabbbbbabbbb"))
"""

#1227. Airplane Seat Assignment Probability

#1154. Day of the Year
"""
def dayOfYear(date):
    leap=[0,31,29,31,30,31,30,31,31,30,31,30,31]
    norm=[0,31,28,31,30,31,30,31,31,30,31,30,31]
    mon=int(date[5:7])
    day=int(date[8:])
    year=int(date[:4])
    if year%4==0:
        if year%100==0:
            if year%400==0:
                return sum([leap[x] for x in range(mon)])+day
            else:
                return sum([norm[x] for x in range(mon)])+day
        else:
            return sum([leap[x] for x in range(mon)])+day
    else:
        print("mon=",mon)
        return sum([norm[x] for x in range(mon)])+day
print(dayOfYear(date = "2019-02-10"))
"""

#1037. Valid Boomerang
"""
def isBoomerang(points):
    if len(set([tuple(x) for x in points]))<3:
        return False
    else:
        d1=[x-y for x, y in zip(points[0],points[1])]
        d2=[x-y for x, y in zip(points[1],points[2])]
        return d1[0]*d2[1]!=d1[1]*d2[0]
print(isBoomerang([[0,0],[2,2],[13,3]]))
"""

#1608. Special Array With X Elements Greater Than or Equal X
"""
def specialArray(nums):
    import collections
    base=collections.Counter(nums)
    res=sorted(list(base.items()), key=lambda x: x[0])[::-1]
    num_count=0
    #if len(res)==1:
     #   return len(nums) if nums[0]>=len(nums) else -1
    for x in range(len(res)):
        k=res[x][0]
        num_count+=res[x][1]
        if k==num_count:
            return k
        elif k>num_count:
            continue
        else:
            if x==0:
                return -1
            else:
                if num_count-res[x][1]>k:
                    return num_count-res[x][1]
                else:
                    return -1
    return len(nums)

print(specialArray(nums = [10,10,10]))
"""

#1491. Average Salary Excluding the Minimum and Maximum Salary
"""
def average(salary):
    return sum(sorted(salary)[1:-1])/(len(salary)-2)
print(average(salary = [6000,5000,4000,3000,2000,1000]))
"""
#1455. Check If a Word Occurs As a Prefix of Any Word in a Sentence
"""
def isPrefixOfWord(sentence, searchWord):
    base=sentence.split()
    alen=len(searchWord)
    for x in range(len(base)):
        if len(base[x])>=alen and base[x][:alen]==searchWord:
            return x+1
    return -1

print(isPrefixOfWord(sentence = "hello from the other side", searchWord = "they"))
"""

#1413. Minimum Value to Get Positive Step by Step Sum
"""
def minStartValue(nums):
    res=0
    tmp=0
    for i in nums:
        tmp+=i
        res=min(res,tmp)
    return -res+1
print(minStartValue(nums = [1,-2,-3]))
"""

#1374. Generate a String With Characters That Have Odd Counts
"""
def generateTheString(n):
    if n==2:
        return "ab"
    if n%2==1:
        return "a"*n
    else:
        if (n/2)%2==0:
            return "a"*(n//2+1)+"b"*(n//2-1)
        else:
            return "a"*(n//2)+"b"*(n//2)
print(generateTheString(16))
"""

#1399. Count Largest Group
"""
def countLargestGroup(n):
    def sumdig(x):
        return sum(map(lambda x: int(x), list(str(x))))
    base=[sumdig(x) for x in range(1,n+1)]
    import collections
    ans=collections.Counter(base)
    key=max(ans.values())   
    return sum([1 for x in list(ans.values()) if x==key])
print(countLargestGroup(15))
"""

#1332. Remove Palindromic Subsequences
def removePalindromeSub(s):
    res=[]
    ans=0
    n=0
    for x in range(len(s)):
        if n==0:
            res.append(s[x])
            n+=1
        else:
            if res[-1]==s[x]:
                res.pop()
                n-=1
                if n==0:
                    ans+=1
            else:
                res.append(s[x])
                n+=1
    return ans+len(res) 
print(removePalindromeSub("ababa"))
    