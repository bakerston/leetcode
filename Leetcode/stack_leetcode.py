#1003. Check If Word Is Valid After Substitutions
"""
def isValid(s):
    st = []
    for ch in s:
        if ch == "c":
            if len(st) < 2 or st[-2:] != ["a", "b"]:
                return -1
            else:
                st.pop()
                st.pop()
        else:
            st.append(ch)
    return not st
print(isValid("aabcbc"))"""

#85. Maximal Rectangle
"""
def maximalRectangle(matrix):
    def getarea(a):
        st = []
        n = len(a)
        r = [0] * (len(a))
        for i, x in enumerate(a):
            if not st or st[-1][1] <= x:
                st.append([i, x])
            else:
                while st and st[-1][1] > x:
                    curi, curx = st.pop()
                    r[curi] = i - curi -1
                st.append([i, x])
        while st:
            curi, curx = st.pop()
            r[curi] = n - curi - 1
        l = [0] * (len(a))
        a = a[::-1]
        for i, x in enumerate(a):
            while st and st[-1][1] > x:
                curi, curx = st.pop()
                l[curi] = i - curi -1
            st.append([i, x])
        while st:
            curi, curx = st.pop()
            l[curi] = n - curi - 1
        ans = 0
        for num, le, ri in zip(a[::-1], l[::-1], r):
            ans = max(ans, num * (le + ri + 1))
        return  ans

    n = len(matrix)
    m = len(matrix[0])
    for i in range(n):
        for j in range(m):
            matrix[i][j] = 1 if matrix[i][j] == "1" else 0
    if n == 1:
        return sum(matrix[0])
    for i in range(1, n):
        for j in range(m):
            matrix[i][j] = matrix[i-1][j] + 1 if matrix[i][j] == 1 else 0
    ans = 0  

    for i in range(n):  
        ans = max(ans, getarea(matrix[i]))
    
    return ans
print(maximalRectangle( matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]))
"""

#1063. Number of Valid Subarrays
"""
def validSubarrays(a):
    st = []
    n = len(a)
    r = [0] * (len(a))
    for i, x in enumerate(a):
        if not st or st[-1][1] <= x:
            st.append([i, x])
        else:
            while st and st[-1][1] > x:
                curi, curx = st.pop()
                r[curi] = i - curi -1
            st.append([i, x])
    while st:
        curi, curx = st.pop()
        r[curi] = n - curi - 1
    return sum(r) + len(r)
print(validSubarrays( [1,4,2,5,3]))
"""
