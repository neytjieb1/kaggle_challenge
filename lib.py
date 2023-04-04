import pandas as pd
import numpy as np
import networkx as nx
from itertools import chain
from scipy import sparse
import math

def SortLabels(GF, nodeindex, labeldict):
    '''Get all neighbours of a node at nodeindex and sort them'''
    neigh = list(GF.neighbors(nodeindex))
    label = [labeldict[i] for i in neigh]
    label = list(chain.from_iterable(label))
    label.sort()
    return label

def GlueLabels(nodelabeldict, index, newlist):
    '''Glue together a list of nodelabels with the node's own label'''
    nodelbl = nodelabeldict[index][0]
    x = ''.join([str(s) for s in newlist])
    return str(nodelbl) + x

def dictToArray(dict):
    '''Helper function'''
    a = np.array(list(dict.items()), dtype=object )
    a[:,1] = list(chain.from_iterable(a[:,1]))
    return a

def draw(G, nc='blue'):
    '''Draw graph with its labels'''
    pos = nx.spring_layout(G, seed=1, k=0.3)
    nx.draw(G, pos, node_color=nc)
    node_labels = nx.get_node_attributes(G, "labels")
    nx.draw_networkx_labels(G, pos, node_labels, font_size=8);

def getnodelblarr(GF):
    '''For a specific graph, generate a (sorted) dataframe of old labels and new glued labels for each node'''
    #Dictionary of labels for graph
    NLbl_dict = nx.get_node_attributes(GF, 'labels')
    #Generate new lbls for each vertex
    newlabels = []
    for v in NLbl_dict:
        sortedlbls = SortLabels(GF, v, NLbl_dict)
        gluedlbl = GlueLabels(NLbl_dict, v, sortedlbls)
        newlabels.append(gluedlbl)

    #Array with columns: node, label, newlabel
    Nlbl_arra0 = np.c_[dictToArray(NLbl_dict), newlabels]
    #And sort
    Nlbl_arra0 = Nlbl_arra0[Nlbl_arra0[:,2].argsort()]
    return Nlbl_arra0

def hashtodic(ALPHAbet, newlblarr, currentmax):
    '''Function to hash newly glued labels; then add the (unique) new ones to the overall alphabet.
    Return a Dataframe with new, old, hashed labels for each node'''
    #Get unique entries
    a = np.unique(newlblarr[:,2]) 
    #Get those that are new also
    a = [a[i] for i in range(len(a)) if a[i] not in ALPHAbet]
    #Hash values
    b = np.arange(len(a))+currentmax
    currentmax = currentmax+len(b)
    dic1 = ALPHAbet | {a[i] : b[i] for i in range(len(a))}
    #relabel
    newlblarr = np.c_[newlblarr, [dic1[newlblarr[:,2][_]] for _ in range(len(newlblarr))]]    
    
    return dic1, pd.DataFrame(newlblarr, columns=["node", 'oldlbl', 'hashed', 'newlbl']), currentmax
 
def assignewlabels(GF, nlblarr):
    '''Helper function: change a graph's labels. (Change directly as working on a copy) '''
    newdict = {nlblarr['node'][i] : [nlblarr['newlbl'][i]] for i in range(len(nlblarr))}
    H = GF.copy()
    nx.set_node_attributes(H, newdict, name='labels')
    return H


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


#generate the Hadamard Matrix
def hadamard(n, dtype=int):

    if n < 1:
        lg2 = 0
    else:
        lg2 = int(math.log(n, 2))
    if 2 ** lg2 != n:
        raise ValueError("n must be an positive integer, and n must be "
                         "a power of 2")

    H = np.array([[1]], dtype=dtype)

    for i in range(0, lg2):
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))

    return H


#count the number of distinct labels in a graph
#def count(arr):
#    vis = []
#    count = 0
#    for i in range(len(arr)):
#        ind = 0
#        for j in range(len(vis)):
#            if arr[i] == vis[j]:
#                ind += 1
#        if ind == 0:
#            count += 1
#            vis.append(arr[i])
#    return np.sort(vis),count


#Hadamard Code Label
def hadamard_label(G,h):
    node_labels = nx.get_node_attributes(G, "labels")
    n = len(node_labels)
    E = range(0,49)
    H = hadamard(2**math.ceil(math.log(50,2)))
    l = np.zeros((n,np.shape(H)[1]))
    
    for i in range(n):
        for j in range(49):

            if node_labels[i][0] == E[j]:
                l[i,:] = H[j,:]
    for _ in range(h):
        for i in range(n):
            l[i,:] = l[i,:] + np.sum(np.array([l[j,:] for j in list(G.neighbors(i))]))
    return(l)


def hadamard_kernel(G1,G2,hmax):
    dist = []
    
    for h in range(hmax):
        counter = 0
        h1 = hadamard_label(G1,h)
        h2 = hadamard_label(G2,h)
        n = np.shape(h1)[0]
        m = np.shape(h2)[0]

        for i in range(n):
            for j in range(m):
                counter += np.linalg.norm(h1[i,:]-h2[j,:],ord=1) 
        dist.append(counter)
    return np.sum(dist)


def weight_matrix(G,edge):
    nodes = sorted(G.nodes())
    size = len(nodes)
    edge_labels = nx.get_edge_attributes(G, "labels")
    adj_mat = np.zeros((size,size))
    for v  in G.edges():
        if edge=='True':
            adj_mat[v[0],v[1]] = edge_labels[(v[0],v[1])][0]
        else:
            adj_mat[v[0],v[1]] = 1
    return adj_mat


def random_walk1(G, u, k):
    node_labels = nx.get_node_attributes(G, "labels")
    curr_node = u
    walk = []
    for i in range(k):
        idx = np.random.randint(0,len(list(G.neighbors(curr_node)))-1)
        curr_node = list(G.neighbors(curr_node))[idx]
        walk.append(node_labels[curr_node])
    return walk


def kernel_randomwalk(G1,G2,edge,prob1=np.repeat(.5,2),prob2=np.repeat(.5,2),c=0.1):

    node_labels1 = nx.get_node_attributes(G1, "labels")
    node_labels2 = nx.get_node_attributes(G2, "labels")
    n1 = len(node_labels1)
    n2 = len(node_labels2)
    
    p1 = np.repeat(prob1[0],n1)
    q1 = np.repeat(prob1[1],n1)
    p2 = np.repeat(prob2[0],n2)
    q2 = np.repeat(prob2[1],n2)
    
    p = np.kron(p1,p2)
    q = np.kron(q1,q2)
    
    A1 = weight_matrix(G1,edge)
    A2 = weight_matrix(G2,edge)
    
    W = np.kron(A1.T,A2.T)

    n = np.shape(W)[0]
    
    V = np.linalg.inv(np.eye(n) - c*W)
    
    return q.T.dot(V).dot(p)

from itertools import product
#def direct_product(G1, G2):
#    GP = nx.Graph()
#    # add nodes
#    for u, v in product(G1, G2):
#       if G1.nodes[u]["labels"] == G2.nodes[v]["labels"]:
#           GP.add_node((u, v))
#            GP.nodes[(u, v)].update({"labels": G1.nodes[u]["labels"]})

    # add edges
#    for u, v in product(GP, GP):
#            if (u[0], v[0]) in G1.edges and (u[1], v[1]) in G2.edges and G1.edges[u[0],v[0]]["labels"] == G2.edges[u[1],v[1]]["labels"]:
#                GP.add_edge((u[0], u[1]), (v[0], v[1]))
#                GP.edges[(u[0], u[1]), (v[0], v[1])].update({"labels":G1.edges[u[0], v[0]]["labels"]})

 #   return GP

#n-th order kernel without using product graph
def kernel_nthorder(G1,G2,edge,n=5):
    A1 = weight_matrix(G1,edge)
    A2 = weight_matrix(G2,edge)
    W = np.kron(A1.T,A2.T)
    s = np.shape(W)[0]
    return np.ones(s).T.dot(W**n).dot(np.ones(s))
    
def kernel_nthorder(DPG,n):
    # G = direct_product(G1,G2)
    A = adjacency_matrix(DPG)
    s = np.shape(A)[0]
    return np.ones(s).T.dot(A**n).dot(np.ones(s))



import scipy.sparse as ss
def DPKernel(DPG):
    adjacencymatrix = nx.adjacency_matrix(DPG)
    rho = ss.linalg.norm(adjacencymatrix)
    infsum = ss.linalg.inv( ss.eye(adjacencymatrix.shape[0], format='csc') - rho*adjacencymatrix )
    k = np.sum(np.abs(infsum))
    return k


def calculateLogits(pred):
    logproba = pred
    logproba[list(np.where(logproba[:,1]==1)[0]), 1] = 1-1e-10
    logit = np.log(logproba[:,1]/(1-logproba[:,1])) 
    print('Range:\n', [np.min(logit), np.max(logit)])
    return logit

def saveDataToFormattedSubmissionFile(predictions, filnename):
    Yte = {'Predicted' : predictions} 
    dataframe = pd.DataFrame(Yte) 
    dataframe.index += 1 
    dataframe.to_csv(filnename, index_label='Id')