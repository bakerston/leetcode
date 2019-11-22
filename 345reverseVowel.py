def reverseVowels( s):

    sList=list(s)
    voList = []
    posList = []
    pos=0
    for pos in range(len(sList)):
        if sList[pos] in ['a','e','i','o','u']:
            posList.append(pos)
            voList.append(sList[pos])
        pos=pos+1
    print(len(voList))
    newVoList = voList[::-1]
    print(len(newVoList))
    for i in range(len(posList)):
        sList[posList[i]] = newVoList[i]
    return ''.join(sList)
print(reverseVowels('hello'))


class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        vowels = set(['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'])
        l = list(s)
        i, j = 0, len(s) - 1
        while i < j:
            while i < j and l[i] not in vowels: i += 1
            while i < j and l[j] not in vowels: j -= 1
            # swap i, j
            if i != j:
                l[i], l[j] = l[j], l[i]
            i, j = i + 1, j - 1
        return ''.join(l)
