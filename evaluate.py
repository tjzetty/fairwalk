"""Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

"""
from gensim.models import Word2Vec

EMBEDDINGS_MODEL = './embeddings.model'

model = Word2Vec.load(EMBEDDINGS_MODEL)

component_indices = connected_components(adj)
component_sizes = np.bincount(component_indices)
components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
nodes_to_keep = [
    idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
]
print("Selecting {0} largest connected components".format(n_components))
print(nodes_to_keep)

