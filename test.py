from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.log import setLogLevel, info
from mininet.cli import CLI

class MyLinearTopo(Topo):
    def __init__(self):
        Topo.__init__(self)

        # Add hosts
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')
        h3 = self.addHost('h3', ip='10.0.0.3/24')

        # Add switches
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')

        # Add links between hosts and switches
        self.addLink(h1, s1)
        self.addLink(h2, s2)
        self.addLink(h3, s3)

        # Add links between switches to form a linear topology
        self.addLink(s1, s2)
        self.addLink(s2, s3)

topos = {'mylineartopo': MyLinearTopo}

def run():
    # 設置日誌級別
    setLogLevel('info')

    # 創建自定義拓撲
    topo = MyLinearTopo()

    # 創建 Mininet 網絡
    net = Mininet(topo=topo,
                  controller=None,  # 我們將手動添加控制器
                  switch=OVSSwitch,  # 使用 OVS 交換機
                  autoSetMacs=True)

    # 添加遠程控制器
    controller = RemoteController('c0', ip='127.0.0.1', port=6653)
    net.addController(controller)

    # 設置交換機使用 OpenFlow 1.4 協議
    for switch in net.switches:
        switch.cmd('ovs-vsctl set bridge %s protocols=OpenFlow14' % switch)

    # 啟動網絡
    net.start()

    # 進入 CLI
    CLI(net)

    # 停止網絡
    net.stop()

if __name__ == '__main__':
    run()
