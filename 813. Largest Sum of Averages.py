def largestSumOfAverages(A, K):
    if K==1:
        return sum(A)/len(A)
    if K==len(A):
        return sum(A)
    A.sort()
    aLen=len(A)
    return int(sum(A[aLen-K+1:])+sum(A[:aLen-K+1])/(aLen-K+1))
print(largestSumOfAverages([9,1,2,3,9],3))