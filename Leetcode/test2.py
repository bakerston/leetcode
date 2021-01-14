
#c=sorted(alist,key=lambda x:[x[0],-x[1]])
#alist=[["a",1],["a",2],["b",1],["b",3],["b",2],["c",1]]
#c=[list(map(lambda x:x[1],g)) for k,g in itertools.groupby(zip(alist,blist),
#cur=[list(g) for k, g in groupby(bloomDay, lambda x: x in t) if not k]
#c=[8  if x==1 else 0 for x in nums]

import bisect
import math
import functools
import collections
import re
import itertools
"""
a=collections.defaultdict(int)
a["A"]=2
a["B"]=3
a["F"]=2
print(a)
print(a["C"])
alist=["A","C","F","B"]
alist.sort(key=lambda x: -a[x])

base1=["A","A","F","F","C"]
a=collections.Counter(base1)
base2=["F"]
b=collections.Counter(base2)
alist=["A","C","F","B"]
alist.sort(key=lambda x: [-a[x],-b[x]])
print(alist)
z=[base1,base2]
print(list(map(collections.Counter(), z)))
"""
a=[1,2,3]
print(any(x==3 for x in a))