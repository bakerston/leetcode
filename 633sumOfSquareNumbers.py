import math

def judgeSquareSum(c):
    return any(int((c - a ** 2) ** .5) ** 2 == int(c - a ** 2) for a in range(1, int(pow(c / 2, 0.5) + 1)))

print(judgeSquareSum(625))
"""def judgeSquareSum(self, c):
    def is_square(N):
        return int(N ** .5) ** 2 == N

    return any(is_square(c - a * a)
               for a in xrange(int(c ** .5) + 1))"""