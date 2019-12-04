import collections

def findPairs(nums, k):
    sameCounter = collections.Counter(nums)

    keyList=list(sameCounter.keys())
    valueList = list(sameCounter.values())
    if k==0:
        #pairList=[keyList[i] for i in range(len(valueList)) if valueList[i]>1]
        return len([keyList[i] for i in range(len(valueList)) if valueList[i]>1])
    elif k<0:
        return 0
    else:
        pair=0
        for pos in range(len(keyList)):
            if int(keyList[pos]+k) in keyList:
                pair+=1
        return pair






print(findPairs([1,2,3,2,3,4,5], 0))