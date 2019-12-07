def compress(chars):
    if len(chars)<=1:
        return len(chars)
    aLen=len(chars)
    for pos in range(aLen-1,0,-1):
        if chars[pos]!=chars[pos-1]
