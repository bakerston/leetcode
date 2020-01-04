def addDigits(self, num):
    """
    :type num: int
    :rtype: int
    """
    numList = [int(x) for x in str(num)]

    if len(numList) == 1:
        return numList[0]

    while len(numList) > 1:
        numSum = sum(numList)
        numList = [int(x) for x in str(numSum)]

    return numList[0]