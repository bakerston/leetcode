def findWords(words):
    return [i for i in words if len(set(map(keyRow,i)))==1]



def keyRow(char):
    keyStr = ["qwertyuiopQWERTYUIOP", "asdfghjklASDFGHJKL", "zxcvbnmZXCVBNM"]
    if char in keyStr[0]:
        return '0'
    if char in keyStr[1]:
        return '1'
    else:
        return '2'

print(findWords(["ase",'alaska','hello']))