def climbStairs(n):
    """
    :type n: int
    :rtype: int
    """
    if n<=2:
        return n
    ways = [0 for i in range(n+1)]
    ways[0] = 1
    ways[1] = 2
    pos=0
    for pos in range(n):
        if ways[pos] == 0:
            ways[pos] = ways[pos - 1] + ways[pos - 2]
    return ways[pos]
print(climbStairs(4))