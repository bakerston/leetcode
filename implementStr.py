def strStr(haystack, needle):
    if len(needle) == 0:
        return 0
    if len(haystack) < len(needle):
        return -1
    pos1 = 0
    match = -1
    for pos1 in range(0, len(haystack) - len(needle) + 1):
        if haystack[pos1] == needle[0]:
            match = pos1
            j = 0
            while j in range(0, len(needle)):
                if haystack[pos1 + j] != needle[j] and match==pos1:
                    match=-1
                else:
                    j = j + 1
            if match==pos1:
                return match
            else:
                match=-1
        else:
            pos1 = pos1 + 1
            match = -1
    return match
print(strStr("h","h"))