def rotateString(A,B):
    if len(A)!=len(B):
        return False
    strList=[A[:i] for i in range(1,len(B))]