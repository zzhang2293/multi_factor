import multiprocessing as mp
from multiprocessing.managers import ListProxy, ValueProxy

class Apple:
    
    def __init__(self, size, applelist):
        self.size = size
        self.applelist = applelist
        self.include = []
    def ripen(self):
        self.size += 1
    
    def change(self):
        self.ripen()
        self.color = "red"
    
    def judge(self, item:str, size:ValueProxy, include:ListProxy):
        if item == 'red':
            include.append(item)
            size.value += 1
        else:
            print("no need change")

    def mainfuc(self):
        manager = mp.Manager()
        size_cp = manager.Value('i', self.size)
        include_cp = manager.list(self.include)
        p_list = []
        for val in self.applelist:
            p = mp.Process(target=self.judge, args=(val, size_cp, include_cp))
            p_list.append(p)
            p.start()
        
        for p in p_list:
            p.join()
        print(include_cp)
        
if __name__ == '__main__':
    apple = Apple( 0, ["red", "yellow", "green", "orange", "red"])
    apple.mainfuc()
    print(apple.include)