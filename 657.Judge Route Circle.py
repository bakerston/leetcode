import collections
def judgeCircle(moves):
    strCounter=collections.Counter(moves)
    return strCounter.get("U")==strCounter.get("D") and strCounter.get("L")==strCounter.get("R")
print(judgeCircle("UDUDDU"))

return moves.count('U') == moves.count('D') and moves.count('L') == moves.count('R')