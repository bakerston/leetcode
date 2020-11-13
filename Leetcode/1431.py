def kidsWithCandies(candies, extraCandies):
        maxcand=max(candies)
        return list(map(lambda x:x+extraCandies>=maxcand,candies))

print(kidsWithCandies([2,3,4,5,6],2))