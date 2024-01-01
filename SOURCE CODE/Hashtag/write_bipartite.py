import networkx as nx
from networkx.algorithms import bipartite

def write_bipartite(G, filename):
	[annotators, tags]= nx.bipartite.sets(G)
	actors = list(sorted(annotators))
	labels = list(sorted(tags))
	A=len(actors)
	L=len(labels)
	edges = sorted(G.edges())
	fp = open(filename,'w')
	fp.write('*vertices '+str(G.number_of_nodes())+'\n')
	for i in np.arange(A):
		fp.write(str(i+1)+' '+actors[i]+' 0.0 0.0 man bipartite 0\n')
	for i in np.arange(L):
		fp.write(str(i+A+1)+' '+labels[i]+' 0.0 0.0 ellipses bipartite 1\n')
	fp.write('*arcs\n')
	fp.write('*edges\n')
	for (a,t) in edges:
		fp.write(str(actors.index(a)+1)+' '+str(labels.index(t)+A+1)+' '+str(G[a][t]['weight'])+'\n')
	fp.close()
		

	