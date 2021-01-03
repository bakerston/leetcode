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
"""
def getModifiedArray(length, updates):
    ans=[0]*(length+1)
    for i,j,n in updates:
        ans[i]+=n
        ans[j+1]-=n
    for i in range(1,len(ans)):
        ans[i]+=ans[i-1]
    return ans[:-1]

print(getModifiedArray(length = 5, updates = [[1,3,2],[2,4,3],[0,2,-2]]))
"""

#6. ZigZag Conversion
"""
def convert(s, numRows):
    if numRows==1 or numRows>=len(s):
        return s
    
    L=['']*numRows
    index,step=0,1

    for x in s:
        L[index]+=x
        if index==0:
            step=1
        elif index==numRows-1:
            step=-1
        index+=step
    return ''.join(L)
"""

#340. Longest Substring with At Most K Distinct Characters
"""
def lengthOfLongestSubstringKDistinct(s, k):
    import collections
    if len(s)==1 or len(collections.Counter(s).keys())<=k:
        return len(s)
    else:
        i,j=0,0
        ans=0
        end=collections.defaultdict(int)
        while j<len(s):
            if len(end.keys())<=k:
                ans=max(ans,j-i)
                end[s[j]]=j
                j+=1
            else:
                tmp=sorted(list(end.items()),key=lambda x:x[1])[0]
                i=tmp[1]+1
                del end[tmp[0]]
        print(end)
        if s[-1] in end.keys() and len(end.keys())<=k:
            ans=max(ans,j-i)
        return ans
print(lengthOfLongestSubstringKDistinct(s = "ab", k = 1))
"""

#878. Nth Magical Number
"""
def nthMagicalNumber(N, A, B):
    def lcm(x, y):
        if x > y:
            greater = x
        else:
            greater = y
        while(True):
            if((greater % x == 0) and (greater % y == 0)):
                lcm = greater
                break
            greater += 1
        return lcm
    if A==B:
        return (N-1)*A
    else:
        if A<B:
            A,B=B,A
        if A%B==0:
            return N*A
        else:
            k=lcm(A,B)
            base=[A*x for x in range(1,k//A)]+[B*y for y in range(1,k//B)]+[0]
            base.sort()
            cnt=k//A+k//B-2

            sec=N//(cnt+1)
            res=N%(cnt+1)
            print(sec,k,cnt,res)
            return k*sec+base[res]
print(nthMagicalNumber(N = 4, A = 2, B = 3
))
"""
#757. Set Intersection Size At Least Two
"""
def intersectionSizeTwo(a):   
    if len(a)==1:
        return 2
    a.sort(key=lambda x:x[1])
    ans=[a[0][-2],a[0][-1]]
    for x in range(1,len(a)):
        if a[x][0]<=ans[-2]:
            continue
        elif a[x][0]<=ans[-1]:
            ans.append(a[x][-1])
        else:
            ans.extend([a[x][-2],a[x][-1]])
    return len(ans)
print(intersectionSizeTwo(a = [[1, 2], [2, 3], [2, 4], [4, 5]])) 
"""

            

#838. Push Dominoes
"""
def pushDominoes(dominoes):
    import collections
    d=collections.deque()
    ans=""
    for x in dominoes:
        if x==".":
            d.append(x)
        elif x=="R":
            if d:
                if d[0]=="R":
                    while d:
                        ans+="R"
                        d.popleft()
                else:
                    while d:
                        ans+=d.popleft()
            d.append(x)
        else:
            if d:
                if d[0]=="R":
                    k=0
                    while d:
                        d.popleft()
                        k+=1
                    if k%2==0:
                        ans+="R"*(k//2)+"."+"L"*(k//2-1)
                    else:
                        ans+="R"*(k//2+1)+"L"*(k//2)
                else:
                    while d:
                        d.popleft()
                        ans+="L"
            d.append(x)
    if d:
        if d[0]=="R":
            while d:
                ans+="R"
                d.popleft()
        elif d[0]==".":
            while d:
                ans+="."
                d.popleft()
        else:
            while d:
                ans+=d.popleft()    
    return ans
print(pushDominoes(".L.R...LR..L.."))
"""

#467. Unique Substrings in Wraparound String
"""
def findSubstringInWraproundString(p):
    d={x:1 for x in p}
    n=len(p)
    cur=1
    for x in range(n-1,-1,-1):
        if x==n-1 or ord(p[x+1])-ord(p[x])!=1:
            cur=1
        else:
            cur+=1
            d[p[x]]=max(d[p[x]],cur)
    return sum(d.values())
print(findSubstringInWraproundString("abcdmcdef"))
"""

#516. Longest Palindromic Subsequence
"""
def longestPalindromeSubseq(s):
    n=len(s)
    dp=[[0]*n for _ in range(n)]
    k=0
    while k<n:
        for i in range(n-k):
            j=i+k
            if i==j:
                dp[i][j]=1
            elif s[i]==s[j]:
                dp[i][j]=dp[i+1][j-1]+2
            else:
                dp[i][j]=max(dp[i+1][j],dp[i][j-1])
        k+=1
    return dp[0][n-1]
print(longestPalindromeSubseq("a"))
"""

#548. Split Array with Equal Sum
"""
def splitArray(nums):
    n=len(nums)
    s=[0]*(n+1)
    for i in range(n):
        s[i+1]=s[i]+nums[i]
    def check(l,r):
        return set(s[m]-s[l] for m in range(l+1,r+1) if s[m]-s[l]==s[r+1]-s[m+1])
    return any(check(0,j-1)&check(j+1,n-1) for j in range(n))

print(splitArray([1,2,1,2,1,2,1]))
"""

#415. Add Strings


#738. Monotone Increasing Digits
"""
def monotoneIncreasingDigits(N):
    s=list(str(N))
    n=len(s)
    ans=""
    cur=0
    for x in range(n):
        if x==0 or s[x]==s[x-1]:
            cur+=1
        elif s[x]>s[x-1]:
            ans+=cur*str(s[x-1])
            cur=1
        else:
            ans+=str(int(s[x-1])-1)+"9"*cur+"".join(s[x+1:])
            return ans
    ans+=str(s[x])*cur
    return  ans
print(monotoneIncreasingDigits(1332))
"""

#522. Longest Uncommon Subsequence II
"""
def findLUSlength(strs):
    def issub(s,l):
        i=0
        for c in l:
            if i<len(s) and s[i]==c:
                i+=1
        return i==len(s)
    
    strs.sort(key=len, reverse=True)
    for i,w1 in enumerate(strs):
        if all (not issub(w1,w2) for j,w2 in enumerate(strs) if i!=j):
            return len(w1)
    return -1
    """
#439. Ternary Expression Parser
"""
def parseTernary(expression):
    st=[]
    n=len(expression)
    for x in reversed(expression):
        st.append(x)
        if len(st)>=2 and st[-2]=="?":
            st[-5:]=st[-3 if st[-1] == 'T' else -5]
    return st
print(parseTernary("F?1:T?4:5"))"""

#1540. Can Convert String in K Moves
"""
def canConvertString(s, t, k):
    import collections
    d=collections.defaultdict(int)
    for x,y in zip(s,t):
        d[(ord(y)-ord(x))%26]+=1
    ans=sorted(d.items(), key=lambda x: [-x[0],-x[1]])[0]
    return ans[0]+26*(ans[1]-1)<=k
print(canConvertString(s = "abc", t = "bcd", k = 10))
"""

#564. Find the Closest Palindrome
"""
def nearestPalindromic(n):
    l=len(n)
    import math
    if n!=n[::-1]:
        a=n[:math.ceil(l/2)]
        print(a)
        return a+a[::-1]
    else:
        if l%2==0:
            if n[l//2-1]=="0":
                return n[:l//2-1]+"11"+n[:l//2-1][::-1]
            else:
                return n[:l//2-1]+str(int(n[l//2-1])-1)*2+n[:l//2-1][::-1]
        else:
            if n[l//2]=="0":
                return n[:l//2]+"1"+n[:l//2][::-1]
            else:
                return n[:l//2]+str(int(n[l//2])-1)+n[:l//2][::-1]
print(nearestPalindromic("1230321"))
"""
#1592. Rearrange Spaces Between Words
"""
def reorderSpaces(text):
    import re
    l=re.split("\s+",text)
    alen=sum(map(len,l))
    sp=len(text)-alen
    word=[x for x in l if x!=""]
    wd=len(word)
    ans=""
    if wd==1:
        ans+=word[0]
        ans+=" "*sp
        return ans
    else:
        sta=sp//(wd-1)
        bac=sp*(wd-1)
        for x in word:
            ans+=x+" "*sta  
        return ans[:len(text)]
print(reorderSpaces( text = "  this   is  a sentence "))
"""

#1323. Maximum 69 Number
"""
def maximum69Number (num):
    i=-1
    for i,x in enumerate(num):
        if x=="6":
            if i==0:
                return "9"+num[1:]
            else:
                return num[:i]+"9"+num[i+1:]
    if len(num)==1:
        return 6
    else:
        return num[:-1]+"6"
print(maximum69Number("999"))
"""
#1427. Perform String Shifts
"""
def stringShift(s, shift):
    r=sum(x[1] for x in shift if x[0]==1)
    l=sum(x[1] for x in shift if x[0]==0)
    n=len(s)
    if abs(r-l)%n==0:
        return s
    elif r>l:
        sh=(r-l)%n
        return s[n-sh:]+s[0:n-sh]
    else:
        sh=(l-r)%n
        return s[sh:]+s[:sh]
print(stringShift(s = "abcdefg", shift = [[1,1],[1,1],[0,2],[1,3]]))
"""

#316. Remove Duplicate Letters
def removeDuplicateLetters(s):
    """
    import collections
    if not s:
        return s
    c=collections.Counter(s)
    used=set()
    stack=[]
    for ch in s:
        c[ch]-=1
        if ch in used:
            continue
        while stack and stack[-1]>ch and c[stack[-1]]>0:
            used.remove(stack.pop())
        stack.append(ch)
        used.add(ch)
    return "".join(stack)

print(removeDuplicateLetters(s = "cbacdcbc"))
"""

#336. Palindrome Pairs
"""
def palindromePairs(words):
    d={x:i for i,x in enumerate(words)}
    ans=[]
    for i in range(len(words)):
        for j in range(len(words[i])+1):
            tmp1=words[i][:j]
            tmp2=words[i][j:]
            if tmp1[::-1] in d and d[tmp1[::-1]]!=i and tmp2==tmp2[::-1]:
                ans.append([i,d[tmp1[::-1]]])
            if j!=0 and tmp2[::-1] in d and d[tmp2[::-1]]!=i and tmp1==tmp1[::-1]:
                ans.append([d[tmp2[::-1]],i])
    return ans
print(palindromePairs(words = ["abcd","dcba","lls","s","sssll"]))
"""

#1698. Number of Distinct Substrings in a String
"""
def countDistinct(s):
    n=len(s)
    ans=0
    for i in range(1,n+1):
        cur=set()
        for st in range(n):
            if st+i<=n and s[st:st+i] not in cur:
                cur.add(s[st:st+i])
                ans+=1
        print(cur)
    return ans
print(countDistinct("aabbaba"))
"""
    
#953. Verifying an Alien Dictionary
"""
def isAlienSorted(words, order):
    d={x:i for i,x in enumerate(order)}
    def oldd(w):
        ans=""
        for x in w:
            ans+=chr(97+d[x])
        return ans
    l=ans=[oldd(x) for x in words]
    return sorted(l)==ans 
print(isAlienSorted(words = ["apple","app"], order = "abcdefghijklmnopqrstuvwxyz"))
"""

#443. String Compression
"""
def compress(chars):
    if len(chars)<2:
        return len(chars)
    i=0
    while i<len(chars)-1:
        cnt=1
        while i<len(chars)-1 and chars[i]==chars[i+1]:
            cnt+=1
            del chars[i]
            print(chars)
        if cnt>1:
            cnt=str(cnt)
            n=len(cnt)
            for j in range(n):
                chars.insert(i+1+j,cnt[j]) 
            i+=n+1
        else:
            i+=1
    print(chars)
    return len(chars)
print(compress(chars = ["a","a","a","a","c","c","c"]))
"""

#811. Subdomain Visit Count
"""
def subdomainVisits(cpdomains):
    import collections
    d=collections.defaultdict(int)
    for c in cpdomains:
        num=c.split(" ")[0]
        st=c.split(" ")[1].split(".")
        for x in range(len(st)):
            tmp=".".join(st[x:])
            d[tmp]+=int(num)
    return [ str(d[i])+" "+i for i in d]
print(subdomainVisits(["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]))
"""

#592. Fraction Addition and Subtraction
"""
def fractionAddition(expression):
    import re
    nu=re.split(r'\+|\-',expression)
    al=re.split(r'(\+|\-)',expression)
    res=1
    for x in nu:
        if x:
            res*=int(re.split(r'/',x)[1])
    ans=0

    if al[0]=="":
        for x in range(2,len(al),2):
            t,d=re.split(r'/',al[x])
            if al[x-1]=="-":
                ans-=int(t)*res//int(d)
            else:
                ans+=int(t)*res//int(d)
    else:

        for x in range(0,len(al),2):
            print(x)
            t,d=re.split(r'/',al[x])
            print(t,d)
            if x>0 and al[x-1]=="-":
                ans-=int(t)*res//int(d)
            else:
                ans+=int(t)*res//int(d)
    def computeGCD(x, y): 
        while(y): 
            x, y = y, x % y 
        return x
    if ans==0:
        return ""
    s=""
    k=computeGCD(ans,res)
    s+=str(ans//k)+"/"+str(res//k)
    return s


print(fractionAddition("2/2-5/3+1/3"))
"""

#1220. Count Vowels Permutation
"""
def countVowelPermutation(n):
    if n==1:
        return 5
    a=e=i=o=u=1
    for x in range(1,n):
        a,e,i,o,u=u+i+e,i+a,e+o,i,i+o
    return (a+e+i+o+u)%(10**9+7)
print(countVowelPermutation(4545))
"""

#68. Text Justification
"""
def fullJustify(words, maxWidth):
    cur,st,i,ans=len(words[0]),0,1,[]
    while i<len(words):
        while i<len(words) and  cur+len(words[i])+1<=maxWidth:
            cur+=len(words[i])+1
            i+=1
        print(st,i)
        if i==len(words):
            tmp=""
            for x in range(st,i):
                tmp+=words[x]+" "
            if len(tmp)>maxWidth:
                ans.append(tmp[:maxWidth])
            else:
                ans.append(tmp+" "*(maxWidth-len(tmp)))
            return ans
        else:
            if i-st==1:
                ans.append(words[st]+""*(maxWidth-len(words[st])))
            else:
                gap=i-st-1
                print(words[st:i])
                sp=maxWidth-sum(map(len,words[st:i]))
                nor=sp//gap
                mor=sp%gap
                tmp=""
                for x in range(mor):
                    tmp+=words[st+x]+" "*(nor+1)
                for x in range(mor,gap):
                    tmp+=words[st+x]+" "*nor
                tmp+=words[st+gap]
                ans.append(tmp)
        st=i
        cur=len(words[st])
        i+=1
    if i==len(words):
        ans.append(words[i-1]+" "*(maxWidth-len(words[st])))
        return ans

print(fullJustify(["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"],
20))
"""

#1163. Last Substring in Lexicographical Order
"""
def lastSubstring(s):
    m="a"
    for x in s:
        if x>m:
            m=x
    ans=""
    for x in range(len(s)):
        if s[x]==m:
            ans=max(ans,s[x:])
    return ans
print(lastSubstring("babzfdsaba"))
"""

#1239. Maximum Length of a Concatenated String with Unique Characters
"""
def maxLength(arr):
    dp=[set()]
    for a in arr:
        if len(set(a))<len(a):
            continue
        a=set(a)
        for c in dp[:]:
            if a&c:
                continue
            dp.append(a|c)
    return max(len(a) for a in dp)
print(maxLength(arr = ["cha","r","act","ers"]))
"""


        

