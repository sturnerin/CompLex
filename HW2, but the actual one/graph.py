import nltk
import networkx as nx
nltk.download('wordnet')
nltk.download('omw')
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
from networkx.algorithms import community


travelset = wn.synset('travel.v.01')
hyponyms = travelset.hyponyms()



lemdict = {}

for hyponym in hyponyms:
    for hyponym2 in hyponyms:
        if hyponym2 != hyponym:
            newlist = []
            newlist.append(hyponym.name())
            newlist.append(hyponym2.name())
            pair = str(sorted(newlist)[0]) + ',' + str(sorted(newlist)[1])
            if pair not in lemdict:
                lemdict[pair] = 0


for key, value in lemdict.items():
    lemm= key.split(',')
    lemm1 = wn.synset(lemm[0])
    lemm2 = wn.synset(lemm[1])

    common = []
    for lang in sorted(wn.langs()):
        lang1 = lemm1.lemma_names(lang)
        lang2 = lemm2.lemma_names(lang)
        for a in lang1:
            for b in lang2:
                if a == b:
                    common.append(a)
    lemdict[key] += len(common)
        
    
G = nx.Graph()
for hyponym in hyponyms:
    G.add_node(hyponym.name())

for key, value in lemdict.items():
    word = key.split(',')
    w1 = word[0]
    w2 = word[1]
    if value > 0:
        G.add_edge(w1, w2, weight=value)

print(G.number_of_edges())
nx.write_gexf(G, 'graph_file.gexf')
pos=nx.spring_layout(G, k=1.5)

edges = G.edges()
weights =[G[u][v]['weight'] for u,v in edges]
nx.draw(G, pos, edges=edges, width=weights)

nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=10) 
nx.draw_networkx_edges(G, pos, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10)
plt.axis('off') 
#plt.show()


print('components - ', nx.number_connected_components(G))
print('коэффициент ассортативности - ', nx.degree_pearson_correlation_coefficient(G))
print('плотность - ', nx.density(G))

communities_generator = community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
print("top level communities:", sorted(map(sorted, top_level_communities)))
#print("next level communities:", sorted(map(sorted, next_level_communities)))

print('another one: \n', community.greedy_modularity_communities(G))
print('aaaaaaand another one: \n', community.kernighan_lin_bisection(G))
