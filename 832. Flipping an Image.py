def flipAndInvertImage( A):
    """
    :type A: List[List[int]]
    :rtype: List[List[int]]
    """
    flipA = [[]] * len(A)
   # print(len(A))
    for i in range(len(A)):
        #print(A[1][::-1])
        flipA[i] = [1 if j == 0 else 0 for j in A[i][::-1]]
    return flipA
print(flipAndInvertImage([[1,1,0,0],[1,0,0,1],[0,1,1,1],[1,0,1,0]]))
