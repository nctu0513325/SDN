from mininet.topo import Topo
from mininet.net import Mininet
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.node import RemoteController

class BroadcastStormTopo(Topo):
    def build(self):
        # 創建 3 個交換機
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')

        # 創建 2 個主機
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')

        # 連接主機到交換機
        self.addLink(h1, s1)
        self.addLink(h2, s3)

        # **建立環狀拓撲 (會造成廣播風暴)**
        self.addLink(s1, s2)
        self.addLink(s2, s3)
        self.addLink(s3, s1)

if __name__ == '__main__':
    topo = BroadcastStormTopo()
    net = Mininet(topo=topo, controller=RemoteController, link=TCLink)

    net.start()
    CLI(net)
    net.stop()
