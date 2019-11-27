from functools import reduce
def constructRectangle(area):
    aset=list(set(reduce(list.__add__, ([i, area // i] for i in range(1, int(pow(area, 0.5) + 1)) if area % i == 0))))
    aset.sort()
    alen=len(aset)
    if alen%2==0:
        return [aset[int(alen/2-1)],aset[int(alen/2)]]
    else:
        return [aset[int((alen-1)/2)],aset[int((alen-1)/2)]]
print(constructRectangle(24))