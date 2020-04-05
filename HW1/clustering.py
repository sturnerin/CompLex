import gensim
import logging
import re
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

def makelist():
    nouns = []
    with open('words.txt', encoding='utf-8') as f:
        words = f.readlines()
    for line in words:
        m = re.search('бросить_.*?\s', line)
        noun = m.group(0)
        noun = noun.replace('бросить_', '')
        noun = noun.replace(' ', '')
        noun = noun.replace('\t', '')
        nouns.append(noun)
    return nouns


def loadmodel(file_name: str):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=True, encoding='utf-8')
    print("word2vec model '%s' loaded" % file_name)
    return model

model = loadmodel('model.bin')
nouns = makelist()



for noun in nouns:
    phrase = 'бросить_' + noun
    noun = noun + '_NOUN'
    v = model['бросить_VERB'] + model[noun]
    vstr = ''
    for element in v:
        vstr = vstr + str(element) + '\t'
    filename = phrase + '.txt'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(vstr)
    with open('all.txt', 'a', encoding='utf-8') as f:
        f.write(vstr)
        f.write('\n')
    



array = []
with open ('all.txt', 'r') as f_input:
    for line in f_input:
        line = line.strip().split('\t')
        vector = []
        for item in line[1:]:
            vector.append(float(item))
        array.append(vector)

X = np.array(array)
X.shape

Z = hcluster.linkage(X)
plt.figure()
dn = hcluster.dendrogram(Z)
#plt.show()

clusters = hcluster.fclusterdata(Z, 1.1)
print('кластеров после иерархической кластеризации: %d' % len(set(clusters)))
print('получилось так: ', clusters)


print('кластеров для к-средних: 10')
kmeans = KMeans(n_clusters=10).fit(X)
print('получилось так: ', kmeans.labels_)
centers = kmeans.cluster_centers_
print('центры: ', centers)
