import matplotlib.pyplot as plt
import networkx as nx

# 单机训练
def plot_single_machine_training():
    G = nx.DiGraph()
    G.add_node("Client")
    nx.draw(G, with_labels=True, arrows=False)
    plt.title("Single Machine Training")
    plt.show()

# 星型网络联邦学习
def plot_star_network():
    G = nx.DiGraph()
    G.add_node("Server")
    for i in range(10):
        client = f"Client {i+1}"
        G.add_node(client)
        G.add_edge(client, "Server")
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.title("Star Network Federated Learning")
    plt.show()

# 环状网络联邦学习
def plot_ring_network():
    G = nx.DiGraph()
    for i in range(10):
        client = f"Client {i+1}"
        G.add_node(client)
        next_client = f"Client {i+2}" if i < 9 else "Client 1"
        G.add_edge(client, next_client)
        G.add_edge(next_client, client)
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.title("Ring Network Federated Learning")
    plt.show()

# 画图
plot_single_machine_training()
plot_star_network()
plot_ring_network()
