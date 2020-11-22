
#146. LRU Cache
"""
class LRUCache:
    def __init__(self, capacity: int):
        self.d={} 
        self.cap=capacity
        self.l=[]
        self.a=0
    def get(self, key: int) -> int:
        if key in self.d.keys():
            return self.d[key]
        else:
            return -1
    def put(self, key: int, value: int) -> None:
        if key in self.d.keys():
            self.d[key]=value
            self.l.remove(key)
            self.l.append(key)
        else:
            if len(self.d.keys())<self.cap:
                self.d[key]=value
                self.l.append(key)
            else:
                self.a=self.l[0]
                self.l.pop(0)
                del self.d[self.a]
                self.l.append(key)
                self.d[key]=value

def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.list = deque(maxlen=capacity)
        self.items = {}
    
    def get(self, key: int) -> int:
        if key not in self.items: return -1
        
        self.list.remove(key) # this adds O(n) time to each get in worst case
        self.list.append(key)
        
        return self.items[key]
    
    def put(self, key: int, value: int) -> None:
        if key in self.items:
            self.list.remove(key)  # this adds O(n) time to each put in worst case
            self.list.append(key)
            self.items[key] = value
            return
        
        if len(self.items) == self.capacity:
            del self.items[self.list.popleft()] # popleft is O(1) thanks to deque, but del from list is O(n) worst case
            
        self.list.append(key)
        self.items[key] = value

"""
#155. Min Stack
"""
class MinStack:
    def __init__(self):

        self.minlist=[]
        self.l=[]

    def push(self, x: int) -> None:
        self.l.append(x)
        if not self.minlist or x<=self.minlist[-1]:
            self.minlist.append(x)

    def pop(self) -> None:
        if not self.l:
            return None
        self.l.pop()
        self.minlist.pop()

    def top(self) -> int:
        if not self.l:
            return None
        return self.l[-1]

    def getMin(self) -> int:
        if not self.minlist:
            return None
        return self.minlist[-1]

a=MinStack()
a.push(-2)
a.push(0)
a.push(-3)
print(a.getMin())
print(a.pop())
print(a.top())
print(a.getMin())
"""

#1381. Design a Stack With Increment Operation
import collections
class UndergroundSystem:

    def __init__(self):
        self.d={}
        self.ave={}

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        self.d[id]=[stationName,t]
            
    def checkOut(self, id: int, stationName: str, t: int) -> None:
        if self.d[id][0] not in self.ave:
            self.ave[self.d[id][0]]={stationName:[t-self.d[id][1]]}
        else:
            if stationName not in self.ave[self.d[id][0]].keys():
                self.ave[self.d[id][0]][stationName]=[t-self.d[id][1]]
            else:
                self.ave[self.d[id][0]][stationName].append(t-self.d[id][1])
        
    def getAverageTime(self, startStation: str, endStation: str) -> float:
        if self.ave[startStation][endStation]:
            return sum(self.ave[startStation][endStation])/len(self.ave[startStation][endStation])
        else:
            return -1
a=UndergroundSystem()
a.checkIn(1,"A",10)
a.checkOut(1,"D",15)
a.checkIn(2,"A",30)
a.checkIn(3,"B",4)
a.checkOut(3,"F",13)
a.checkOut(2,"D",120)
print(a.getAverageTime("A","D"))