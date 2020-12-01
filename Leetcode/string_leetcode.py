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
"""
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
"""

#1525. Number of Good Ways to Split a String
"""
def numSplits(s):
    import collections
    r=collections.Counter(s)
    l=collections.defaultdict(int)
    p=0
    q=len(r.keys())
    ans=0
    for x in s:
        print(x)
        r[x]-=1
        l[x]+=1
        if r[x]==0:
            q-=1
        if l[x]==1:
            p+=1      
        if p==q:
            ans+=1
    return ans
print(numSplits(s = "acbadbaada"))
"""

#1493. Longest Subarray of 1's After Deleting One Element
"""
def longestSubarray(nums):
    import itertools
    base=[list(y) for x,y in itertools.groupby(nums)]
    if len(base)==1:
        return 0 if nums[0]==0 else len(nums)-1
    elif len(base)==2:
        return len(base[0]) if base[0][0]==1 else len(base[1])
    elif len(base)==3:
        if base[0][0]==0:
            return len(base[1])
        else:
            return len(base[0])+len(base[2]) if len(base[1])<=1 else max(len(base[0]),len(base[2]))
    else:
        if nums[0]==0:
            ans=len(base[1])
            for i in range(3,len(base),2):
                if len(base[i-1])>=2:
                    ans=max(ans,len(base[i]))
                else:
                    ans=max(ans,len(base[i])+len(base[i-2]))
        else:
            ans=len(base[0])
            for i in range(2,len(base),2):
                if len(base[i-1])>=2:
                    ans=max(ans,len(base[i]))
                else:
                    ans=max(ans,len(base[i])+len(base[i-2]))
    return ans
print(longestSubarray( nums = [1,1,0,1]))
"""

#831. Masking Personal Information
"""
def maskPII(S):
    import re
    if "@" in S:
        c=re.split("@|\.",S)
        ans=c[0][0].lower()+"*"*5+c[0][-1].lower()+"@"+c[1].lower()+"."+c[2].lower()
        return ans
    else:
        c=[x for x in S if ord(x)<=57 and ord(x)>=48]
        ans=""
        if len(c)>10:
            return "+"+"*"*(len(c)-10)+"-"+"*"*4+"-"+"*"*4+"-"+"".join(c[-4:])
        else:
            return "*"*4+"-"+"*"*4+"-"+"".join(c[-4:])
print(maskPII("1(234)567-890"))
"""

#921. Minimum Add to Make Parentheses Valid
"""
def minAddToMakeValid(S):
    l,r=0,0
    for x in S:
        if x ==")":
            if l==0:
                r+=1
            else:
                l-=1
        else:
            l+=1
    return l+r
print(minAddToMakeValid(")))())(())"))
"""

#916. Word Subsets
"""
def wordSubsets(A, B):
    import collections
    count = collections.Counter()
    for b in B:
        count=count | collections.Counter(b)
    return [a for a in A if collections.Counter(a)&count ==count ]

print(wordSubsets(A = ["amazon","apple","facebook","google","leetcode"], B = ["ec","oc","ceo"]))
"""

#1078. Occurrences After Bigram
"""
def findOcurrences(text,first, second):
    if len(text.split())<=2:
        return []
    else:
        ans=[]
        base=text.split()
        for x in range(len(base)-2):
            if base[x]==first and base[x+1]==second:
                ans.append(base[x+2])
        return ans
print(findOcurrences(text = "alice is a good girl she is a good student", first = "a", second = "good"))
"""

#1641. Count Sorted Vowel Strings
#unsolved


#1347. Minimum Number of Steps to Make Two Strings Anagram
"""
def minSteps(s, t):
    import collections
    s_cnt=collections.Counter(s)
    t_cnt=collections.Counter(t)
    ans=0
    for x in t_cnt.keys():
        if s_cnt[x]:
            ans+=min(s_cnt[x],t_cnt[x])
    return len(s)-ans
print(minSteps(s = "anagram", t = "mangaar"))
"""

#890. Find and Replace Pattern
"""
def findAndReplacePattern(words, pattern):
    import collections
    def getpost(pat):
        d=collections.defaultdict(list)
        for x in range(len(pat)):
            d[pat[x]].append(x)
        return d
    return [x for x in words if list(getpost(x).values())==list(getpost(pattern).values())]


print(findAndReplacePattern(words = ["abc","deq","mee","aqq","dkd","ccc"], pattern = "abb"))
"""

#1314. Matrix Block Sum
"""
def matrixBlockSum(mat, K):

    m,n=len(mat),len(mat[0])
    base=[[0]*(n+1) for x in range(m+1)]
    for i in range(m):
        for j in range(n):
            base[i+1][j+1]=base[i+1][j]+base[i][j+1]-base[i][j]+mat[i][j]
    ans=[[0]*n for x in range(m)]
    for i in range(m):
        for j in range(n):
            left=max(0,j-K)
            right=min(n,j+1+K)
            top=max(0,i-K)
            bot=min(m,i+1+K)
            ans[i][j]=base[bot][right]-base[top][right]-base[bot][left]+base[top][left]
    return ans

    m, n = len(mat), len(mat[0])
    rangeSum = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            rangeSum[i + 1][j + 1] = rangeSum[i + 1][j] + rangeSum[i][j + 1] - rangeSum[i][j] + mat[i][j]
    ans = [[0] * n for _ in range(m)]        
    for i in range(m):
        for j in range(n):
            r1 = max(0, i - K)
            c1 = max(0, j - K)
            r2 = min(m, i + K + 1)
            c2 = min(n, j + K + 1)
            ans[i][j] = rangeSum[r2][c2] - rangeSum[r1][c2] - rangeSum[r2][c1] + rangeSum[r1][c1]
    return ans

    
print(matrixBlockSum(mat = [[1,2,3],[4,5,6],[7,8,9]], K = 1))
"""
#1111. Maximum Nesting Depth of Two Valid Parentheses Strings
"""
def maxDepthAfterSplit(seq):
    n=0
    cur=0
    for x in seq:
        if x=="(":
            cur+=1
            n=max(cur,n)
        else:
            cur-=1
    sp=n//2
    ans=[]
    cur=0
    for x in seq:
        if x=="(":
            if cur>=sp:
                ans.append(1)
            else:
                ans.append(0)
            cur+=1
        else:
            if cur>sp:
                ans.append(1)
            else:
                ans.append(0)
            cur-=1
    return ans
print(maxDepthAfterSplit(seq = "(()())"))
"""

#1147. Longest Chunked Palindrome Decomposition
"""
def longestDecomposition(text):
    res=0
    l=""
    r=""
    for x,y in zip(text,text[::-1]):
        l=l+x
        r=y+r
        if l==r:
            res+=1
            l=""
            r=""  
    return res
print(longestDecomposition(text = "ghiabcdefhelloadamhelloabcdefghi"))
"""

#1513. Number of Substrings With Only 1s
"""def numSub(s):
    import itertools
    base=[len(list(g)) for k,g in itertools.groupby(s) if k=="1"]
    return sum([n*(n+1)//2 for n in base])
print(numSub("0110111"))
"""

#1589. Maximum Sum Obtained of Any Permutation
"""
def maxSumRangeQuery(nums, requests):
    n=len(nums)
    cnt=[0]*(n+1)
    for i,j in requests:
      cnt[i]+=1
      cnt[j+1]-=1
    for x in range(1,n+1):
        cnt[x]+=cnt[x-1]
    print(cnt)
    return sum([x*y for x,y in zip(sorted(cnt[:-1]),sorted(nums))])%(10**9+7)


print(maxSumRangeQuery(nums = [1,2,3,4,5], requests =[[1,3],[0,1]]))
"""

#1616. Split Two Strings to Make Palindrome
"""
def checkPalindromeFormation(a,b):
    def cnmsb(a,b):
        if a==a[::-1] or b==b[::-1]:
            return True
        n=len(a)
        done=True
        for x in range(n//2):
            if a[x]!=b[-x-1]:
                done=False
                break
        if done:
            return True
        else:
            if x==0:
                return False
            else:
                return a[x:-x]==a[x:-x][::-1] or b[x:-x]==b[x:-x][::-1]
    return cnmsb(a,b) or cnmsb(b,a)
    
print(checkPalindromeFormation(a = "adfb", b = "bbma"))
"""       
#1233. Remove Sub-Folders from the Filesystem
"""
def removeSubfolders(folder):
    folder.sort()
    res=list()
    res.append(folder[0])
    for x in folder[1:]:
        if x.find(res[-1]+"/")==-1:
            res.append(x)
    return res
print(removeSubfolders(folder = ["/a/b/c","/a/b/c/ds","/a/b/ca","/a/b/d"]))

"""

#1234. Replace the Substring for Balanced String
"""
def balancedString(s):
    import collections
    base=collections.Counter(s)
    more=set([x for x in base.keys() if base[x]>len(s)//4])
    k=sum([base[x] for x in more])-len(more)*len(s)//4
    i,j=0,0
    alen=0
    ans=len(s)
    while j<=len(s):
        if alen<k:
            alen+= s[j] in more
            ans+=1
        elif alen==k:


 

    return k
print(balancedString("QQEWREWRQQQQ"))
"""

#249. Group Shifted Strings
"""
def groupStrings(strings):
    import collections
    d=collections.defaultdict(list)
    ans=[]
    ones=[]
    for x in strings:
        if len(x)==1:
            ones.append(x)
        else:
            tmp=[(ord(x)-ord(y))%26 for x,y in zip(x[:-1],x[1:])]
            ans.append([x,tuple(tmp)])
    for x in ans:
        d[x[1]].append(x[0])
    return list(d.values())+[ones]
print(groupStrings(["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"]))
"""

#161. One Edit Distance
"""
    if len(s)==len(t):
        for i in range(len(t)):
            if s[i]!=t[i]:
                return s[i+1]==t[i+1]
        return False
    else:
        if len(s)<len(t):
            s,t=t,s
        for i in range(len(t)):
            if s[i]!=t[i]:
                return t[i:]==s[i+1:]
        return True
print(isOneEditDistance(s = "", t = "A"))
"""

#370. Range Addition
def getModifiedArray(length, updates):
    ans=[0]*(length+1)
    for i,j,n in updates:
        ans[i]+=n
        ans[j+1]-=n
    for i in range(1,len(ans)):
        ans[i]+=ans[i-1]
    return ans[:-1]

print(getModifiedArray(length = 5, updates = [[1,3,2],[2,4,3],[0,2,-2]]))