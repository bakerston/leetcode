def validPalindrome(s):
    if len(s)<=2:
        return True
    num=0
    start = 0
    end = len(s) - 1
    while end-start>=1:
        if s[start] != s[end]:
            if end<len(s)-1:
                a=check(s[start+1:end+1])
                b=check(s[start:end])
            else:
                a=check(s[start+1:])
                b=check(s[start:end])
            return a==True or b==True
        start+=1
        end-=1
    return True

def check(s):
    print(s)
    start=0
    end=len(s)-1
    if len(s)==1:
        return True
    if len(s)==2:
        return s[0]==s[1]
    while end-start>=2:
        if s[start]!=s[end]:
            return False
        end-=1
        start+=1
    return True
print(validPalindrome("aabb"))
