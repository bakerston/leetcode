def hammingDistance(x,y):

    #a="{0:b}".format(x)
    #b="{0:b}".format(y)

    c= x^y
    clist=list(bin(c))
    blist=[i for i in clist if i=='1']
    #return len([i for i in list(bin(c)) if i==1])
    return len(blist)
print(hammingDistance(1,4)

return bin(x ^ y).count('1')