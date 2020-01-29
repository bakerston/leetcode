def lengthOfLastWord( s):
    """
    :type s: str
    :rtype: int
    """
    if len(s)==1:
        if s[0]==" ":
            return 0
        else:
            return 1
    alen = len(s)
    pos = len(s) - 1
    while pos >= 0:
        if s[pos] != " ":
            lenWord = 1
            pos = pos - 1
            while s[pos] != " " and pos>=0:
                pos = pos - 1
                lenWord += 1
            return lenWord
        pos = pos - 1
    return False
print(lengthOfLastWord("  a "))