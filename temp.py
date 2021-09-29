import networkx as nx
import matplotlib.pyplot as plt
G = nx.Graph()
layers = [2,5,2]
w0 = [[-0.5719,  0.6129],[ 0.6422, -0.6959],[-0.3211,  0.5527],[ 0.4624,  0.9387],[ 0.9747, -0.0043]]
w1 = [[ 0.8226,  0.2435,  0.9319,  0.9443, -0.6634], [ 0.3555, -0.1458,  0.1979, -0.9345,  0.1461]]
b0 = [ 0.1330, -0.3916, -0.5924,  0.1491,  0.6028]
b1 = [-0.1323, -0.3572]
weights = [w0,w1]
biases = [b0,b1]

nodes = [list() for _ in range(len(layers))]
node_cnt = 0
for layer_number in range(len(layers)):
    layer_size = layers[layer_number]
    for i in range(layer_size):
        nodes[layer_number].append(node_cnt)
        G.add_node(node_cnt)
        node_cnt += 1

for layer_number in range(1, len(layers)):
    prev_layer = nodes[layer_number-1]
    layer = nodes[layer_number]
    for node_ind in range(len(layer)):
        for prev_node_ind in range(len(prev_layer)):
            node = layer[node_ind]
            prev_node = prev_layer[prev_node_ind]
            weight = weights[layer_number-1][node_ind][prev_node_ind]
            G.add_edge(prev_node,node, weight=weight)

max_y = 40.0

positions = dict()
for i in range(len(layers)):
    current_y = 0.0
    dy = max_y / len(nodes[i])
    for j in range(len(nodes[i])):
        positions[nodes[i][j]] = (i, current_y)
        current_y += dy

edges = G.edges()
weights = [G[u][v]['weight'] for u,v in edges]

nx.draw(G, pos = positions, width=weights)
# labels = nx.get_edge_attributes(G,'weight')
# nx.draw_networkx_edge_labels(G,positions,edge_labels=labels)
plt.show()
