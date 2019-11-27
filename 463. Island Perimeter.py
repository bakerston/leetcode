def islandPerimeter(grid):
    num=0
    adj=0
    for i in range(len(grid[0])):
        for j in range(len(grid)):
            if grid[j][i]==1:
                num+=1
                if j>0 and grid[j-1][i]==1:
                    adj+=1
                if j<len(grid)-1 and grid[j+1][i]==1:
                    adj+=1
                if i>0 and grid[j][i-1]==1:
                    adj+=1
                if i<len(grid[0])-1 and grid[j][i+1]==1:
                    adj+=1
    return int(4*num-adj)
print(islandPerimeter([[0,1,0,0],
 [1,1,1,0],
 [0,1,0,0],
 [1,1,0,0]]
))
