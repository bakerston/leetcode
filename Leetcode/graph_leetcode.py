#361. Bomb Enemy
def maxKilledEnemies(grid):
    n=len(grid)
    m=len(grid[0])
    l,r,u,d=[[-1]*m for _ in range(n)], [[-1]*m for _ in range(n)], [[-1]*m for _ in range(n)], [[-1]*m for _ in range(n)]   
    for i in range(n):
        cur=0
        for j in range(m):
            if grid[i][j]=="W":
                cur=0
            elif grid[i][j]=="E":
                cur+=1
            #grid[i][j]=="0"
            else:
                l[i][j]=cur
    for i in range(n):
        cur=0
        for j in range(m-1,-1,-1):
            if grid[i][j]=="W":
                cur=0
            elif grid[i][j]=="E":
                cur+=1
            #grid[i][j]=="0"
            else:
                r[i][j]=cur
    for j in range(m):
        cur=0
        for i in range(n):
            if grid[i][j]=="W":
                cur=0
            elif grid[i][j]=="E":
                cur+=1
            #grid[i][j]=="0"
            else:
                u[i][j]=cur
    for j in range(m):
        cur=0
        for i in range(n-1,-1,-1):
            if grid[i][j]=="W":
                cur=0
            elif grid[i][j]=="E":
                cur+=1
            #grid[i][j]=="0"
            else:
                d[i][j]=cur
    ans=0
    for x in range(n):
        for y in range(m):
            if grid[x][y]=="0":
                ans=max(ans,l[x][y]+r[x][y]+u[x][y]+d[x][y])
    return ans
print(maxKilledEnemies([["0","E","0","0"],["E","0","W","E"],["0","E","0","0"]]))

