def test(s, k):
    if k <= 0:
        return s
    aLen = len(s)
    if aLen < k:
        return s[::-1]
    mul, res = divmod(aLen, k)
    strList = []
    for i in range(1,mul+1):
        strList+=[s[(i-1)*k:i*k]]
    strList+=[s[mul*k:]]
    #=[s[::-1] for s in strList if ]
    for j in range(len(strList)):
        if j%2==0:
            strList[j]=strList[j][::-1]

    return ''.join(strList)

print(test("adsasgacasewtq",3))

for idx in range(0, len(s), 2*k):
            s = s[:idx] + s[idx:idx+k][::-1] + s[idx+k:]
        return s

