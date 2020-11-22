#944. Delete Columns to Make Sorted
"""
def minDeletionSize(A):
    import functools
    base=[functools.reduce(lambda x,y:x+y,[m[x] for m in A]) for x in range(len(A[0]))]
    c=[1 for a in base if list(a)==sorted(a)]
    return len(A[0])-sum(c)


print(minDeletionSize( A = ["zyx","wvu","tsr"]))
"""

#1288. Remove Covered Intervals
"""
def removeCoveredIntervals(intervals):
    intervals.sort(key=lambda x: x[0])

    tmp=intervals[0]
    cnt=1
    for i in range(1,len(intervals)):
        cur=intervals[i]
        if cur[1]>tmp[1]:
            cnt+=1
        tmp[0]=max(tmp[0],cur[0])
        tmp[1]=max(tmp[1],cur[1])
    return cnt
print(removeCoveredIntervals(intervals = [[1,2],[3,5],[5,8],[6,8]]))
"""