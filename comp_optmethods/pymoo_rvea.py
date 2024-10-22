#!/usr/bin/env python
import numpy as np
import networkx as nx
import math
import time
import pickle
import argparse
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.rvea import RVEA


import simple_graph
import HiberniaGlobal_graph, Colt_graph, Funet_graph, Cogent_graph, Abvt_graph, Intellifiber_graph, DialtelecomCz_graph, TataNld_graph, Kdl_graph, Internode_graph, Missouri_graph, Ion_graph, Ntelos_graph, UsCarrier_graph, Palmetto_graph

class ONOSControllerPlacement(ElementwiseProblem):
    def __init__(self, num_nodes, distance_matrix, shortest_paths, graph, **kwargs):
        super().__init__(n_var=2*num_nodes, 
                         n_obj=4, 
                         n_constr=2, 
                         xl=0, xu=1, 
                         **kwargs)
        self.num_nodes = num_nodes
        self.distance_matrix = distance_matrix
        self.shortest_paths = shortest_paths
        self.graph = graph
    
    def _evaluate(self, x, out, *args, **kwargs):
        controller_nodes = x[:self.num_nodes]   # first half is controller placement
        atomix_nodes = x[self.num_nodes:]       # second half is atomix placement


        num_controller = np.sum(controller_nodes)
        num_atomix = np.sum(atomix_nodes)

        # Obj1: Minimize number of contrtoller
        f1 = num_controller

        # Obj2: Minimize number of atomix
        f2 = num_atomix

        # Obj3: Minimize average FSP
        f3 = calculate_FST(self.num_nodes, 
                           controller_nodes, 
                           atomix_nodes, 
                           self.distance_matrix, 
                           self.shortest_paths)
        
        f4 = calculate_BC(self.num_nodes, 
                           controller_nodes, 
                           atomix_nodes, 
                           self.distance_matrix, 
                        #    self.shortest_paths,
                           self.graph)

        # Constr1: The number of controller is equal to or greater than 2
        g1 = 2 - num_controller

        # Constr2: The number of atomix is equal to or greater than 3
        g2 = 3 - num_atomix
        
        # Add the centrality metrix into optimazing objectives:
        # 1. Nearest controller for each switch
        # 2. The number of controlled switches for each controller should be <= limit_num_switches_controlled (limit_num_switches_controlled=int(math.ceil(num_nodes/num_controller)))
        # 3. return value should be the variance for all controller's betweenness centrality
        out["F"] = [f1, f2, f3, f4]
        out["G"] = [g1, g2]


def calculate_FST(num_nodes, controller_nodes, atomix_nodes, distance_matrix, shortest_paths):
    num_controller = np.sum(controller_nodes)
    num_atomix = np.sum(atomix_nodes)
    controller_list = np.nonzero(controller_nodes)[0].tolist()
    atomix_list = np.nonzero(atomix_nodes)[0].tolist()

    if(num_controller == 0 or num_atomix ==0):
        return math.inf

    # find the nearest controller for each switch
    controller_of = []
    for s in range(num_nodes):
        delay = math.inf
        nearest_controller = None
        for c in controller_list:
            if distance_matrix[s][c] < delay:
                delay = distance_matrix[s][c]
                nearest_controller = c
        controller_of.append(nearest_controller)    

    # calculate average delay to atomix nodes from each controller
    average_atomix_delay_from = {}
    for c in controller_list:
        delay = []
        for a in atomix_list:
            delay.append(distance_matrix[c][a])
        average_atomix_delay_from[c] = np.mean(delay)

    # find the nearest atomix for each atomix and calculate average delay
    atomix_atomix_delays = []
    for a1 in atomix_list:
        delay = math.inf
        for a2 in atomix_list:
            if(a1 == a2):
                continue
            if distance_matrix[a1][a2] < delay:
                delay = distance_matrix[a1][a2]
        atomix_atomix_delays.append(delay)
    average_atomix_atomix_delay = np.mean(atomix_atomix_delays)
    FTSs = []
    for source in range(num_nodes):
        for distination in range(num_nodes):
            if(source == distination):
                continue
            delay = 0
            is_controlled_by_single_controller = True
            counted_controllers = []
            for s in shortest_paths[source][distination]:
                # switch-controller delay
                delay += distance_matrix[s][controller_of[s]] * 4

                # controller-atomix delay
                if(s == source):
                    delay += average_atomix_delay_from[controller_of[s]] * 2
                elif(s != distination):
                    if(controller_of[s] != controller_of[source]):
                        is_controlled_by_single_controller = False
                        if(not controller_of[s] in counted_controllers):
                            counted_controllers.append(controller_of[s])
                            delay += average_atomix_delay_from[controller_of[s]]
                else:
                    if(controller_of[s] == controller_of[source]):
                        if(not is_controlled_by_single_controller):
                            delay += average_atomix_delay_from[controller_of[s]]
                    else:
                        delay += average_atomix_delay_from[controller_of[s]] * 2
            
            # atomix-atomix delay
            delay +=  average_atomix_atomix_delay * 2
            FTSs.append(delay)

    return np.mean(FTSs)



def calculate_BC(num_nodes, controller_nodes, atomix_nodes, distance_matrix, graph):
    G = nx.Graph()
    for node1 in range(len(graph)):
        G.add_node(str(node1))
        for node2, delay in graph[node1].items():
            G.add_edge(str(node1), str(node2), weight=delay)
    
    # The list of betweenness centrality for all switches
    nodes_bc=nx.current_flow_betweenness_centrality(G, normalized=True, weight=None, dtype='float', solver='full')
    num_controller = np.sum(controller_nodes)
    num_atomix = np.sum(atomix_nodes)
    controller_list = np.nonzero(controller_nodes)[0].tolist()

    if(num_controller == 0 or num_atomix ==0):
        return math.inf

    # find the nearest controller for each switch
    controller_of = []
    limit_num_switches_controlled=int(math.ceil(num_nodes/num_controller)) # balance the number of switches controllers can control 
    switches_bc_of_controller_ = dict.fromkeys((range(num_nodes)),0) # list of sum of betweenness centrality of switches for each controller
    for s in range(num_nodes):
        delay = math.inf
        nearest_controller = None
        controlled_switches=[]
        for c in controller_list:
            # Conditions: nearest controller (with the lowest delay) && the number of switches for each controller < limit_num_switches_controlled
            if distance_matrix[s][c] < delay and controller_of.count(c) < limit_num_switches_controlled:
                delay = distance_matrix[s][c]
                nearest_controller = c
                controlled_switches.append(s)
        switches_bc_of_controller_[nearest_controller] += nodes_bc[str(s)]
        controller_of.append(nearest_controller)
    
    # Simplify switches_bc_of_controller_ (only need value for calculating variance)
    bc_array = []
    for i in switches_bc_of_controller_.values():
        bc_array.append(i)

    # return variance value can show the degree of balance within all controllers
    return np.var(bc_array)

def calc_distance_matrix(graph):
    G = nx.Graph()
    for vertex, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(vertex, neighbor, weight=weight)
    distance_matrix = dict(nx.all_pairs_dijkstra_path_length(G))
    shortest_paths = dict(nx.all_pairs_dijkstra_path(G))

    return distance_matrix, shortest_paths

def optimize(graph):
    num_nodes = len(graph)
    pop_size=100
    distance_matrix, shortest_paths = calc_distance_matrix(graph)

    problem = ONOSControllerPlacement(num_nodes, distance_matrix, shortest_paths, graph)
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=6)
    algorithm = RVEA(pop_size=pop_size,
                      ref_dirs=ref_dirs,
                      sampling=BinaryRandomSampling(),
                      crossover=TwoPointCrossover(),
                      mutation=BitflipMutation(),
                      eliminate_duplicates=True
                      )

    start = time.time()
    res = minimize(problem,
               algorithm,
               termination=('n_gen', 1000),
               seed=1,
               save_history=True,
               verbose=True)
    end = time.time()
    dt = end - start
    print(f'RVEA - Elapsed time: {dt}')
    print('Total number of nodes: ', num_nodes)

    return res

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='Prefix for the result file')
    return parser.parse_args()

def main():
    args = parse_arg()

    # res = optimize(simple_graph.graph)
    # res = optimize(Cogent_graph.graph)
    # res = optimize(UsCarrier_graph.graph)
    # res = optimize(HiberniaGlobal_graph.graph)
    # res = optimize(Colt_graph.graph)
    # res = optimize(Funet_graph.graph)
    # res = optimize(Abvt_graph.graph)
    # res = optimize(Intellifiber_graph.graph)
    # res = optimize(DialtelecomCz_graph.graph) # with node id problem
    # res = optimize(TataNld_graph.graph)
    # res = optimize(Kdl_graph.graph)
    # res = optimize(Internode_graph.graph)
    # res = optimize(Missouri_graph.graph)
    # res = optimize(Ion_graph.graph)
    # res = optimize(Ntelos_graph.graph) # with node id problem
    res = optimize(Palmetto_graph.graph)

    F = res.F
    print(F[np.argsort(F[:, 0])])

    # outputfile = 'res_bc_simple_rvea.pkl'
    # outputfile = 'res_bc_Cogent_rvea.pkl'
    # outputfile = 'res_bc_UsCarrier_rvea.pkl'
    # outputfile = 'res_bc_HiberniaGlobal_rvea.pkl'
    # outputfile = 'res_bc_Colt_rvea.pkl'
    # outputfile = 'res_bc_Funet_rvea.pkl'
    # outputfile = 'res_bc_Abvt_rvea.pkl'
    # outputfile = 'res_bc_Intellifiber_rvea.pkl'
    # outputfile = 'res_bc_DialtelecomCz_rvea.pkl'
    # outputfile = 'res_bc_TataNld_rvea.pkl'
    # outputfile = 'res_bc_Kdl_rvea.pkl'
    # outputfile = 'res_bc_Internode_rvea.pkl'
    # outputfile = 'res_bc_Missouri_rvea.pkl'
    # outputfile = 'res_bc_Ion_rvea.pkl'
    # outputfile = 'res_bc_Ntelos_rvea.pkl'
    outputfile = 'res_bc_Palmetto_rvea.pkl'

    if(args.prefix):
        outputfile = args.prefix + outputfile
    with open(outputfile, 'wb') as f:
        pickle.dump(res,f)


if __name__ == "__main__":
    main()