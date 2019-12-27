import collections
def findAnagrams(s, p):
    """
    :type s: str
    :type p: str
    :rtype: List[int]
    """
    if not p or not s or len(s)<len(p):
        return False

    pCounter = collections.Counter(p)
    anaList = []
    pLen=len(p)
    sLen=len(s)
    pos=0
    while pos <(sLen - pLen + 1):
        sCounter=collections.Counter(s[pos:pos+pLen])
        if sCounter==pCounter:
            anaList.append(pos)
            while pos+pLen<sLen and s[pos] == s[pos + pLen]:
                anaList.append(pos + 1)
                pos = pos + 1
        pos=pos+1
    return anaList

print(findAnagrams('abca;lksdjlashdgbjncdbladsfjasbdcdefgcbd','dbc'))

"""if len(p) == 0:
    return []
if len(s) < len(p):
    return []
if len(s) == len(p):
    if sorted(s) == sorted(p):
        return [0]
    else:
        return []
pList = sorted(p)
anaList = []
for pos in range(len(s) - len(p) + 1):
    if sorted(s[pos:pos + len(p)]) == pList:
        anaList.append(pos)
return anaList"""
