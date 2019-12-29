def numJewelsInStones(J, S):
    num=0
    for pos in range(len(S)):
        if S[pos] in J:
            num+=1
    return num
print(numJewelsInStones("zF","afzffF"))