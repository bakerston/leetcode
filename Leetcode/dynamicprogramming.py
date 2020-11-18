#1105. Filling Bookcase Shelves
def minHeightShelves(books, shelf_width):
    n=len(books)
    dp=[float('inf')]*(n+1)
    print(dp)
    dp[0]=0
    for i in range(1,n+1):
        max_width=shelf_width
        max_height=0
        j=i-1
        while j>=0 and max_width>=books[j][0]:
            max_width-=books[j][0]
            max_height=max(max_height,books[j][1])
            dp[i]=min(dp[i],dp[j]+max_height)
            j-=1
    print(dp)
    return dp[n]
print(minHeightShelves(books = [[1,1],[2,3],[2,3],[1,1],[1,1],[1,1],[1,2]], shelf_width = 4))
