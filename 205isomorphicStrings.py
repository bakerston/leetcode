def isIsomorphic(s,t):
    charLen=len(s)
    dic={}
    for pos in range(len(s)):
        if not s[pos] in dic.keys() and not t[pos] in dic.values():
            dic[s[pos]]=t[pos]
        elif not s[pos] in dic.keys() and  t[pos] in dic.values():
            return False
        elif s[pos] in dic.keys() and not t[pos] in dic.values():
            return False
        elif dic.get(s[pos])!=t[pos]:
            return False
    return True
print(isIsomorphic('aba','baa'))

class Solution:
    def isIsomorphic(self, s1, s2):
        return len(set(zip(s1, s2))) == len(set(s1)) == len(set(s2))