def isToeplitzMatrix(matrix):
    dic = {}
    row = len(matrix)
    col = len(matrix[0])
    for i in range(row):
        for j in range(col):
            if i - j not in dic.keys():
                dic[i -j]=matrix[i][j]
            else:
                if dic[i - j] != matrix[i][j]:
                    return False
    return True
print(isToeplitzMatrix([[1,2],[2,2]]))