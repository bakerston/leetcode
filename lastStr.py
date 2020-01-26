def lengthOfLastWord( s):
    """
    :type s: str
    :rtype: int
    """
    s1 = reversed(s)
    Found = False
    alen = len(s1)
    pos = 0
    while pos < alen and not Found:
        if s1[pos] in "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPADSFGHJKLZXCVBN":
            starter = pos
            newPos = pos
            while s1[newPos] in "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPADSFGHJKLZXCVBN":
                newPos=newPos+1
            ender=newPos
        return ender-starter
    return 0

print(lengthOfLastWord("12eg3gs3"))