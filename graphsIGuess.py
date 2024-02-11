import itertools

import matplotlib.pyplot as plt
import networkx as nx
from scipy.io import mmread

file_path = 'power-1138-bus.mtx'

# Read the Matrix Market file using scipy
sparse_matrix = mmread(file_path)

# Convert the sparse matrix to a NetworkX graph
graph = nx.convert_matrix.from_scipy_sparse_array(sparse_matrix)
self_loops = list(nx.selfloop_edges(graph))
graph.remove_edges_from(self_loops)

# Function to find communities in the graph
def find_communities(graph):
    resolution = 0.12  # Example value, you can adjust this

    # Find communities with adjusted resolution
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph, resolution=resolution))
    return communities


# Function to find most important nodes based on different centralities
def find_most_important_nodes(graph):
    centralities = {
        "Degree Centrality": nx.degree_centrality(graph),
        "Betweenness Centrality": nx.betweenness_centrality(graph),
        "Closeness Centrality": nx.closeness_centrality(graph),
    }
    most_important_nodes = {}
    for centrality_name, centrality_dict in centralities.items():
        most_important_node = max(centrality_dict, key=centrality_dict.get)
        most_important_nodes[centrality_name] = most_important_node
    return most_important_nodes


def visualize_graph(graph, communities, community=True):
    print(communities.__sizeof__())
    pos = {}

    important_nodes = find_most_important_nodes(graph)

    node_size = 150

    pos.update(nx.random_layout(graph))

    # Assign initial positions using spring layout
    if community:

        # Assign x-coordinate positions based on community
        x_offset = 0
        for community in communities:
            for node in community:
                pos[node] = (pos[node][0] + x_offset, pos[node][1])
            x_offset += 1.25  # Adjust this value to control the separation between communities

        # Draw the graph
        colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

        for community, color in zip(communities, colors):
            node_sizes = [node_size * 4 if node in important_nodes.values() else node_size for node in community]
            nx.draw_networkx_nodes(graph, pos, nodelist=community, node_color=color, node_size=node_sizes,
                                   edgecolors='black')
        nx.draw_networkx_edges(graph, pos)
    else:
        node_sizes = [node_size * 4 if node in important_nodes.values() else node_size for node in graph.nodes()]
        nx.draw(graph, pos=pos, with_labels=False, node_size=node_sizes)

    node_labels = {node: str(node) if node in important_nodes.values() else '' for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos=pos, labels=node_labels)

# Find communities in the graph
communities = find_communities(graph)

# Find most important nodes based on different centralities
most_important_nodes = find_most_important_nodes(graph)

o = 0
for centrality_name, node in most_important_nodes.items():
    plt.text(-0.85, 1.23 - o, f"Highest {centrality_name}: Node {node}", fontsize=13, color='black')
    o += 0.05

visualize_graph(graph, communities, community=True)

# Display the plot
plt.title("Power network divided in communities")
plt.show()



degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
degree_count = nx.degree_histogram(graph)

# Plot the degree distribution
plt.bar(range(len(degree_count)), degree_count, width=0.8, color='r')

# Add labels and title
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution')

for i, count in enumerate(degree_count):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.xticks(range(len(degree_count)))

# Display the plot
plt.show()