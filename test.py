"""ef binary_search_bisect(lst, x):
    from bisect import bisect_left
    i = bisect_left(lst, x)
    if i != len(lst) and lst[i] == x:
        return i
    return None

def searchRange( nums, target):
    l=binary_search_bisect(nums,target)
    if l==None:
        return [-1,-1]
    if l==len(nums)-1:
        return [l,l]
    if nums[-1]==nums[l]:
        return [l,len(nums)-1]
    end=len(nums)-1
    start=l
    while end-start>1 and nums[end]>nums[start]:
        mid=int((end+start)/2)
        if nums[mid]==nums[l]:
            start=mid
            print(start)
        else:
            end=mid
            print(end)

    return [l,start]
print(searchRange([0,0,2,2,2,2,2,3,3,3,3,3],0))

def spiralOrder(matrix):
    res=[]
    top,bot,left,right=0,len(matrix)-1,0,len(matrix[0])-1
    while (bot>top and right==left) or (bot==top and right>left) or(bot>top and right>left):
        outer(matrix,top,bot,left,right,res)
        top+=1
        bot-=1
        left+=1
        right-=1
    return res

def outer(matrix,top,bot,left,right,res):
    if right==left and top==right:
        res.append(matrix[top][left])"""

"""
def rob(nums):
    if len(nums)==1:
        return nums[0]
    elif len(nums)==2:
        return max(nums)
    elif len(nums)==3:
        return max(nums[1],nums[0]+nums[2])
    else:
        three_ago=nums[0]
        two_ago=nums[1]
        one_ago=nums[2]+nums[0]
        for day in range(3,len(nums)):
            today=max(three_ago,two_ago)+nums[day]
            three_ago,two_ago,one_ago=two_ago,one_ago,today
        return max(one_ago,two_ago)
print(rob([2,1,1,2]))
a=10004
test_list=["1","2","3"]
result=map(lambda x: str(a).find(x),test_list)
print(list(result)==[0,-1,-1])"""

"""def backspaceCompare(S, T):

    return backspaceStr(S) == backspaceStr(T)


def backspaceStr(astr):
    my_stack = []
    for i in range(len(astr)):
        if astr[i] == "#" and len(my_stack) != 0:
            my_stack = my_stack[:-1]
            print(my_stack)
        elif astr[i] == "#" and len(my_stack) == 0:
            print(my_stack)
            continue
        else:
            my_stack.append(astr[i])
            print(my_stack)
    res = ""
    while len(my_stack) != 0:
        res += my_stack[-1]
        my_stack = my_stack[:-1]

    return res"""

"""import operator
alist=["20:20","21:19","17:20","3:03"]
def split(astr):
    a,b=astr.split(":")
    return (int(a),int(b))
blist=list(map(split,alist))
print(blist)
sorted(blist,key=operator.itemgetter(0,1))
print(blist)"""

print(sum(map(ord,"daf")))