import collections
def intersect(nums1, nums2):
    interList=[]
    count1=collections.Counter(nums1)
    count2=collections.Counter(nums2)
    keyList=[i for i in count1.keys() if i in count2.keys()]
    for key in keyList:
        interList+=[key]*min(count1.get(key),count2.get(key))
    return interList
print(intersect([ 4, 1, 17,28, 28, 26, 66, 91],[9, 9, 74, 21, 45, 11, 63]))