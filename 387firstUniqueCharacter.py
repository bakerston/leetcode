import collections


def firstUniqChar(s):

    """sList = [i for i in s]
    strCounter = collections.Counter(sList)
    vList = list(strCounter.values())
    kList = list(strCounter.keys())
    print(vList)
    print(kList)
    if not 1 in vList:
        return -1
    for pos in range(len(vList)):
        if vList[pos] == 1:
            return str(sList.index(kList[pos]))"""
    sList = [i for i in s]
    strCounter = collections.Counter(sList)
    vList = list(strCounter.values())
    kList = list(strCounter.keys())
    if not 1 in vList:
        return -1
    for pos in range(len(vList)):
        if vList[pos] == 1:
            return sList.index(kList[pos])

print(firstUniqChar("loveleetcode"))
