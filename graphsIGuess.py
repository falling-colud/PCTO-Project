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
    resolution = 0.12

    # Find communities with adjusted resolution
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph, resolution=resolution))
    return communities


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
    }
    most_important_nodes = {}
    for centrality_name, centrality_dict in centralities.items():
        most_important_node = max(centrality_dict, key=centrality_dict.get)
        most_important_nodes[centrality_name] = most_important_node
    return most_important_nodes


def visualize_graph(graph):
    plt.title("Power network")

    nx.draw_spring(graph, with_labels=False, node_size=80, edgecolors='black')


def visualize_graph_communities(graph, communities):
    print(communities.__sizeof__())
    pos = {}

    important_nodes = find_most_important_nodes(graph)

    node_size = 120

    # Assign initial positions using spring layout
    pos.update(nx.random_layout(graph))

    plt.title("Power network divided in communities")

    # Assign x-coordinate positions based on community
    x_offset = 0
    for community in communities:
        for node in community:
            pos[node] = (pos[node][0] + x_offset, pos[node][1])
        x_offset += 1.25  # Adjust this value to control the separation between communities

    # Draw the graph
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for community, color in zip(communities, colors):
        node_sizes = [node_size * 7 if node in important_nodes.values() else node_size for node in community]
        nx.draw_networkx_nodes(graph, pos, nodelist=community, node_color=color, node_size=node_sizes,
                               edgecolors='black')
    nx.draw_networkx_edges(graph, pos)

    node_labels = {node: str(node) if node in important_nodes.values() else '' for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos=pos, labels=node_labels)

    o = 0
    for centrality_name, node in important_nodes.items():
        plt.text(-0.65, 1.23 - o, f"{centrality_name}: Node {node}", fontsize=13, color='black')
        o += 0.05


def visualize_graph_slp(graph):

    longest_shortest_path = find_longest_shortest_path(graph)
    longest_shortest_path_graph = graph.subgraph(longest_shortest_path)

    plt.title("Power network and diameter.\nLength:" + str(len(longest_shortest_path)) + "nodes")

    # Draw the graph
    pos = nx.spring_layout(graph)  # You can choose a different layout if needed
    nx.draw(graph, pos, with_labels=False, node_color='lightblue', edge_color='black', node_size=35)
    nx.draw_networkx_edges(graph, pos, edgelist=longest_shortest_path_graph.edges(), edge_color='red', width=2)
    nx.draw_networkx_nodes(graph, pos, nodelist=longest_shortest_path, node_color='red', node_size=500,
                           edgecolors='black')
    node_labels = {node: longest_shortest_path.index(node) + 1 if node in longest_shortest_path else '' for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos=pos, labels=node_labels)


dy = 0.01
formatter = '{:.1f}'

def visualize_degree_frequency(graph):
    degree_centrality = nx.degree_histogram(graph)

    # Plot the degree distribution

    plt.plot(range(len(degree_centrality)), degree_centrality, "r-.o")

    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')


def visualize_betweenness_frequency(graph):
    betweenness_centrality = nx.betweenness_centrality(graph)

    d = {}
    i = dy
    for centrality in sorted(betweenness_centrality.values()):
        if centrality < i:
            key = formatter.format(i*10) + "-" + formatter.format((i + dy)*10)
            d[key] = d.get(key, 0) + 1
        else:
            i += dy

    values = sorted(d.values())
    values.reverse()

    keys = d.keys()

    # Plot the degree distribution

    plt.plot(keys, values, "g-.o")
    # Add labels and title
    plt.xlabel('Betweennes')
    plt.ylabel('Frequency')
    plt.title('Betweennes Distribution')

def visualize_closeness_frequency(graph):
    closeness_centrality = nx.closeness_centrality(graph)

    d = {}
    i = dy
    for centrality in sorted(closeness_centrality.values()):
        if centrality < i:
            key = formatter.format(i*10) + "-" + formatter.format((i + dy)*10)
            d[key] = d.get(key, 0) + 1
        else:
            i += dy

    values = sorted(d.values())
    values.reverse()

    keys = d.keys()

    # Plot the degree distribution

    plt.plot(keys, values, "b-.o")
    # Add labels and title
    plt.xlabel('Closeness')
    plt.ylabel('Frequency')
    plt.title('Closeness Distribution')


def visualize_sp_frequency(graph):
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


visualize_graph(graph)
plt.show()

#Find communities in the graph
communities = find_communities(graph)
visualize_graph_communities(graph, communities)
plt.show()

visualize_graph_slp(graph)
plt.show()

visualize_sp_frequency(graph)
plt.show()

visualize_betweenness_frequency(graph)
plt.show()

visualize_closeness_frequency(graph)
plt.show()

visualize_degree_frequency(graph)
plt.show()


