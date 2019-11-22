import collections
def numberOfBoomerangs( points):
    sumA=0

    for i in range(len(points)):
        disList=[getDistance(points[i],points[j]) for j in range(len(points)) if j!=i]
        counter=collections.Counter(disList)
        blist=list(counter.values())
        clist=[i*(i-1) for i in blist]
        sumA+=sum(clist)
    return sumA


def getDistance(i, j):
    return (i[0]-j[0])**2 +(i[1]-j[1])**2

print(numberOfBoomerangs([[0,1],[0,2],[2,0],[1,1],[2,2]]))