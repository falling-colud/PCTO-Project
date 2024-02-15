import itertools

import matplotlib.pyplot as plt
import networkx as nx
from scipy.io import mmread

file_path = 'power-494-bus.mtx'

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


def node_with_most_shortest_paths(graph):
    node_counts = {node: 0 for node in graph.nodes()}  # Initialize counts for all nodes

    # Iterate over all pairs of nodes
    for source in graph.nodes():
        for target in graph.nodes():
            if source != target:
                # Calculate shortest path
                shortest_path = nx.shortest_path(graph, source=source, target=target)

                # Increment count for each node in the shortest path
                for node in shortest_path:
                    node_counts[node] += 1

    # Find the node with the highest count
    max_count = max(node_counts.values())
    node_with_max_count = {node: 1 for node, count in node_counts.items() if count == max_count}

    return node_with_max_count

def find_longest_shortest_path(graph):
    final_shortest_path = []

    # Iterate over all pairs of nodes
    for source in graph.nodes():
        for target in graph.nodes():
            if source != target:
                # Calculate shortest path
                shortest_path = nx.shortest_path(graph, source=source, target=target)

                if len(shortest_path) > len(final_shortest_path):
                    final_shortest_path = shortest_path

    # Find the node with the highest count

    return final_shortest_path

def find_longest_shortest_path(graph):
    final_shortest_path = []

    # Iterate over all pairs of nodes
    for source in graph.nodes():
        for target in graph.nodes():
            if source != target:
                # Calculate shortest path
                shortest_path = nx.shortest_path(graph, source=source, target=target)

                if len(shortest_path) > len(final_shortest_path):
                    final_shortest_path = shortest_path

    # Find the node with the highest count

    return final_shortest_path

# Function to find most important nodes based on different centralities
def find_most_important_nodes(graph):
    centralities = {
        "Highest Degree Centrality": nx.degree_centrality(graph),
        "Highest Betweenness Centrality": nx.betweenness_centrality(graph),
        "Highest Closeness Centrality": nx.closeness_centrality(graph),
        "Most Shortest Paths": node_with_most_shortest_paths(graph),
    }
    most_important_nodes = {}
    for centrality_name, centrality_dict in centralities.items():
        most_important_node = max(centrality_dict, key=centrality_dict.get)
        most_important_nodes[centrality_name] = most_important_node
    return most_important_nodes

def visualize_graph(graph):
    plt.title("Power network")
    pos = {}

    important_nodes = find_most_important_nodes(graph)

    node_size = 150

    pos.update(nx.spring_layout(graph))

    node_sizes = [node_size * 4 if node in important_nodes.values() else node_size for node in graph.nodes()]
    nx.draw(graph, pos=pos, with_labels=False, node_size=node_sizes, edgecolors='black')

    node_labels = {node: str(node) if node in important_nodes.values() else '' for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos=pos, labels=node_labels)

    o = 0
    for centrality_name, node in most_important_nodes.items():
        plt.text(-0.65, 1.23 - o, f"{centrality_name}: Node {node}", fontsize=10, color='black')
        o += 0.03


def visualize_graph_communities(graph, communities):
    print(communities.__sizeof__())
    pos = {}

    important_nodes = find_most_important_nodes(graph)

    node_size = 120

    pos.update(nx.random_layout(graph))

    # Assign initial positions using spring layout
    plt.title("Power network divided in communities")

    # Assign x-coordinate positions based on community
    x_offset = 0
    for community in communities:
        for node in community:
            pos[node] = (pos[node][0] + x_offset, pos[node][1])
        x_offset += 1.25  # Adjust this value to control the separation between communities

    # Draw the graph
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

    for community, color in zip(communities, colors):
        node_sizes = [node_size * 7 if node in important_nodes.values() else node_size for node in community]
        nx.draw_networkx_nodes(graph, pos, nodelist=community, node_color=color, node_size=node_sizes,
                               edgecolors='black')
    nx.draw_networkx_edges(graph, pos)

    node_labels = {node: str(node) if node in important_nodes.values() else '' for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos=pos, labels=node_labels)

    o = 0
    for centrality_name, node in most_important_nodes.items():
        plt.text(-0.65, 1.23 - o, f"{centrality_name}: Node {node}", fontsize=10, color='black')
        o += 0.03

def visualize_graph_slp(graph):
    plt.title("Power network and diameter")

    longest_shortest_path = find_longest_shortest_path(graph)
    print(longest_shortest_path.__sizeof__())
    longest_shortest_path_graph = graph.subgraph(longest_shortest_path)

    # Draw the graph
    pos = nx.spring_layout(graph)  # You can choose a different layout if needed
    nx.draw(graph, pos, with_labels=False, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(graph, pos, edgelist=longest_shortest_path_graph.edges(), edge_color='red', width=2)
    nx.draw_networkx_nodes(graph, pos, nodelist=longest_shortest_path, node_color='blue', node_size=500, edgecolors='black')

# Find most important nodes based on different centralities
most_important_nodes = find_most_important_nodes(graph)

visualize_graph_slp(graph)

# Display the plot
plt.show()

# Find communities in the graph
communities = find_communities(graph)

visualize_graph_communities(graph, communities)

# Display the plot
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

plt.show()
# Display the plot
all_shortest_paths = nx.shortest_path_length(graph)

# Count the frequency of each shortest path length
shortest_path_lengths = {}
for source, targets in all_shortest_paths:
    for target, length in targets.items():
        if length not in shortest_path_lengths:
            shortest_path_lengths[length] = 0
        shortest_path_lengths[length] += 1

# Plot the frequency of shortest path lengths
plt.bar(shortest_path_lengths.keys(), shortest_path_lengths.values())
plt.xlabel('Shortest Path Length')
plt.ylabel('Frequency')
plt.title('Frequency of Shortest Path Lengths')
plt.show()

