def sumSubarrayMins(A):
    alen=len(A)
    ans=sum(A)
    if alen==0:
        return 0
    while alen>0:
        res=getdiff(A)
        ans+=sum(res)
        A=res
        alen-=1
    return ans%(10**9+7)
def getdiff(alist):
    if len(alist)==1:
        return []
    ans=[]
    for x in range(len(alist)-1):
        ans.append(min(alist[x],alist[x+1]))
    return ans
print(sumSubarrayMins([3,1,2,4,0,1,2]))