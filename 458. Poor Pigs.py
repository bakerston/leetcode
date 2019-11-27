def poorPigs(buckets, minutesToDie, minutesToTest):
    if minutesToDie>minutesToTest:
        return False
    group=int(minutesToTest/minutesToDie)
    bucketsForGroup=int(buckets/group)+1 if buckets%group!=0 else int(buckets/group)
    return len(bin(bucketsForGroup))-2
print(poorPigs(1000,16,60))