def surfaceArea(grid):
    num=0
    height=0
    ans=0
    for i in grid:
        height+=sum(i)
        num+=sum([1 for x in i if x!=0])

    for i in range(len(grid)):

    =list(map(lambda x,y:abs(x-y),tmpo[:-1],tmpo[1:]))
        print(alist)
        ans+=sum(alist)
        alist=[]
    tmp=zip(*grid)
    print(list(tmp))
    for i in tmp:
        alist = map(lambda x,y:min(x,y),i[:-1],i[1:])
        print(list(alist))
        ans+=sum(alist)
    print(num,height,ans)
    return num*2+height*4-ans*2

print(surfaceArea([[1,2],[3,4]]
))
