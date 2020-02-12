def merge(nums1,m,nums2,n):
    nums1=[nums1[i] for i in range(m)]
    nums1.extend(nums2)
    nums1.sort()
    return nums1
print(merge([1,12,23,0,0,0],3,[4,12,56],3))
