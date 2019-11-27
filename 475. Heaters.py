def findRadius(houses,heaters):



def findminDis(self,aint,alist):
    if aint in alist:
        return 0
    elif aint < alist[0]:
        return int(alist[0] - aint)
    elif aint > alist[-1]:
        return int(aint - alist[-1])
    else:
        alen = len(alist)
        start = 0
        end = alen - 1
        mid = int(alen / 2)
        while end - start >= 1:
            if