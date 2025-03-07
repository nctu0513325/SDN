from mininet.topo import Topo

class LinearTopo(Topo):
    def build(self, k=3):
        # 創建 k 個交換機
        switches = [self.addSwitch('s%d' % (i + 1)) for i in range(k)]
        
        # 創建 k 個主機並連接到對應的交換機
        for i in range(k):
            host = self.addHost('h%d' % (i + 1))
            self.addLink(host, switches[i])
        
        # 連接交換機形成線性拓撲
        for i in range(k - 1):
            self.addLink(switches[i], switches[i + 1])

topos = {'linear': LinearTopo}
