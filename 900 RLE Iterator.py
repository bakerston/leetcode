import collections
class RLEIterator:
    def __init__(self, A):
        self.num = collections.deque()
        self.time = collections.deque()
        for i in range(int(len(A) / 2)):
            self.num.append(A[2*i + 1])
            self.time.append(A[2*i])
    def next(self, n):
        while len(self.time):
            while n > 0:
                tmp_time = self.time.popleft()
                tmp_num = self.num.popleft()
                if n > tmp_time:
                    if not len(self.time):
                        return -1
                    n -= tmp_time
                elif n==tmp_time:
                    return tmp_num
                else:
                    res = tmp_time - n
                    self.time.appendleft(res)
                    self.num.appendleft(tmp_num)
                    print(self.time,self.num)
                    return tmp_num
        return -1
obj=RLEIterator([3,8,0,9,2,5])
param=[2,1,1,2]
for i in param:
    print(obj.next(i))
