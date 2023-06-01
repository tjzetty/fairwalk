import numpy as np
import networkx as nx
from fairwalk import FairWalk

# FILES
EMBEDDING_FILENAME = './embeddings.emb'
EMBEDDING_MODEL_FILENAME = './embeddings.model'
EDGES_EMBEDDING_FILENAME = './edge-embeddings.emb'

# Create a graph
graph = nx.fast_gnp_random_graph(n=100, p=0.5)
n = len(graph.nodes())
node2group = {node: group for node, group in zip(graph.nodes(), (5*np.random.random(n)).astype(int))}
nx.set_node_attributes(graph, node2group, 'group')

# Calculate metrics before fairness
print(graph)
print("Average Degree: ", sum(dict(graph.degree()).values()) / len(graph))
print("LCC: ", graph.subgraph(sorted(nx.connected_components(graph), key=len, reverse=True)[0]))
print("Triangle Count: ", sum(nx.triangles(graph).values()))

# Precompute probabilities and generate walks
model = FairWalk(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
print(model.graph)

# Embed
model = model.fit(window=10, min_count=1,
                  batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the FairWalk constructor)

# Output loss metric
print("Loss: {}".format(model.get_latest_training_loss()))

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)
print("Model {} has been saved.".format(EMBEDDING_MODEL_FILENAME))

# Embed edges using Hadamard method
from fairwalk.edges import HadamardEmbedder

edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

# Look for embeddings on the fly - here we pass normal tuples
print(edges_embs[('1', '2')])
''' OUTPUT
array([ 5.75068220e-03, -1.10937878e-02,  3.76693785e-01,  2.69105062e-02,
... ... ....
...................................................................],
dtype=float32)
'''

# Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
edges_kv = edges_embs.as_keyed_vectors()

# Look for most similar edges - this time tuples must be sorted and as str
# edges_kv.most_similar(str(('1', '2')))

# Save embeddings for later use
edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)