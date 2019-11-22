def findContentChildren(g, s):
    g.sort()
    s.sort()
    gLen=len(g)
    sLen=len(s)
    if sLen==0 or gLen==0:
        return 0
    posG=0
    posS=0
    maxNum=0
    while posG in range(gLen) and posS in range(sLen):
        if g[posG]<=s[posS]:
            maxNum+=1
            posG+=1
            posS+=1
        else:
            posS+=1
    return maxNum
print(findContentChildren([1,2,3],[1,1,23,434,567]))
