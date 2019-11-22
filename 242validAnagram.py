def isAnagram(s, t):
    sList=list(s)
    tList=list(t)
    sAscii=[ord(i) for i in sList]
    tAscii=[ord(i) for i in tList]
    sAscii.sort()
    tAscii.sort()
    return sAscii==tAscii

print(isAnagram("car","rac"))


def isAnagram1(self, s, t):
    dic1, dic2 = {}, {}
    for item in s:
        dic1[item] = dic1.get(item, 0) + 1
    for item in t:
        dic2[item] = dic2.get(item, 0) + 1
    return dic1 == dic2


def isAnagram2(self, s, t):
    dic1, dic2 = [0] * 26, [0] * 26
    for item in s:
        dic1[ord(item) - ord('a')] += 1
    for item in t:
        dic2[ord(item) - ord('a')] += 1
    return dic1 == dic2


def isAnagram3(self, s, t):
    return sorted(s) == sorted(t)

#O(n)
def isAnagram(self, s, t):
    maps = {}
    mapt = {}
    for c in s:
        maps[c] = maps.get(c, 0) + 1
    for c in t:
        mapt[c] = mapt.get(c, 0) + 1
    return maps == mapt