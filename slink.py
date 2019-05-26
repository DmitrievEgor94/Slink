import math
import collections
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

number_of_elements = 10

data_set = np.random.standard_normal((number_of_elements, 1))


def distance(x, y):
    return np.linalg.norm(x - y)

P = []
L = []

for i in range(number_of_elements):
    P.append(i)
    L.append(math.inf)
    M = []

    for j in range(i):
        M.append(distance(data_set[i], data_set[j]))

    for j in range(i):
        if L[j] >= M[j]:
            M[P[j]] = min(M[P[j]], L[j])
            L[j] = M[j]
            P[j] = i
        else:
            M[P[j]] = min(M[P[j]], M[j])

    for j in range(i):
        if L[j] >= L[P[j]]:
            P[j] = i

indexes = list(range(number_of_elements - 1))
P_L_i = list(zip(P, L, indexes))
P_L_i = sorted(P_L_i, key=lambda x: x[1])

map_index_cluster = collections.defaultdict(int)
num_els_in_cluster = collections.defaultdict(int)

for i in range(number_of_elements):
    map_index_cluster[i] = i
    num_els_in_cluster[i] = 1

linkage_matrix = []

new_cluster_index = number_of_elements

P, L, indexes = [x[0] for x in P_L_i], [x[1] for x in P_L_i], [x[2] for x in P_L_i]


for p, l, index in P_L_i:
    clust_index_1 = map_index_cluster[index]
    clust_index_2 = map_index_cluster[p]

    row = [clust_index_1, clust_index_2, l, num_els_in_cluster[clust_index_1]+num_els_in_cluster[clust_index_2]]
    linkage_matrix.append(row)

    map_index_cluster[p] = new_cluster_index
    num_els_in_cluster[new_cluster_index] = num_els_in_cluster[clust_index_1]+num_els_in_cluster[clust_index_2]

    new_cluster_index += 1


plt.figure()
dn = hierarchy.dendrogram(linkage_matrix)
plt.show()
