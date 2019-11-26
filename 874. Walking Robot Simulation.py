def new21Game(N, K, W):
    lower=max(K-W,0)
    endtest,unqutest=0,0
    print(lower)
    for i in range(lower,K):
       # print(posi(W,i))
        endtest+=posi(W,i)*(W-(K-i)+1)
       # print(W-(K-i)+1)
        unqual=max(0,i+W-N)

        unqutest+=posi(W,i)*unqual
    print(endtest,unqutest)
    return 1-unqutest/endtest

def posi(maxi,target):
    tmp=[0 for x in range(target+1)]
    tmp[0]=1
    for i in range(1,target+1):
        ans=0
        lower=max(i-maxi,0)
        for j in range(lower,i):
            ans+=tmp[j]
        tmp[i]=ans
    return tmp[-1]

print(new21Game(21,17,10))