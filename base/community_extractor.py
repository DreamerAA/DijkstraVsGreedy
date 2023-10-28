import networkx as nx
from typing import List, Any
import numpy as np

class CommunityExtractor:
    def __init__(self, net:nx.Graph):
        self.net = net

    @staticmethod
    def extractUseIterator(iter):
        result = []
        while True:
            # item will be "end" if iteration is complete
            item = next(iter, False)
            if item == False:
                break
            result.append(item)
        return result

    def girvan_newman(self, count:int)->List[List[Any]]:
        communities_generator = nx.community.girvan_newman(self.net)
        for _ in range(count-2):
            next(communities_generator)

        result = sorted(map(sorted, next(communities_generator)))
        return result

    def fluid_communities(self, count:int)->Any:
        iter = nx.community.asyn_fluidc(self.net, count)
        return CommunityExtractor.extractUseIterator(iter)
    
    def label_propagation(self)->Any:
        iter = nx.community.asyn_lpa_communities(self.net)    
        return CommunityExtractor.extractUseIterator(iter)
    
    def greedy_modularity(self, resolution:float = 0.5):
        return nx.community.naive_greedy_modularity_communities(self.net, resolution)
    
    def louvain_communities(self):
        return nx.community.louvain_communities(self.net)
    
    def graph_with_community(self, communities):
        count_nodes = len(self.net.nodes())
        num_to_type = np.zeros(shape=(count_nodes,),dtype=int)
        for i,community in enumerate(communities):
            for n in community:
                num_to_type[n] = i
        ngraph = nx.Graph()
        ngraph.add_nodes_from([(n, {"type_id": num_to_type[n]}) for n in self.net.nodes()])
        ngraph.add_edges_from(self.net.edges())
        return ngraph