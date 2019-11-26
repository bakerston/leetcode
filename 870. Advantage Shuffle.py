def advantageCount(A, B):
    A=sorted(A)
    b=sorted(B)
    for i in range(len(b)):
        for j in range(len(A)):
            if A[j]>B[i]