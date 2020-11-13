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
def balancedStringSplit(s):
    mark=s[0]
    tmp=1
    ans=0
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