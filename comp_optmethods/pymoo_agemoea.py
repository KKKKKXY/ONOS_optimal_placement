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
from pymoo.algorithms.moo.age import AGEMOEA

import simple_graph

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
        out["G"] = [g1,g2]


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
    avr_min_a_to_ela_delays = []
    if len(atomix_list) ==1: 
        min_avr_atomix_atomix_delay = distance_matrix[atomix_list[0]][atomix_list[0]]
    else: # only add the aveage dalay of math.ceil(num_atomix/2) parts
        for a1 in atomix_list:
            delay = math.inf
            a_to_ela_delays = [] # the collection of delays from a1 to a2 (a1 != a2)
            for a2 in atomix_list:
                if(a1 == a2):
                    continue
                else:
                    a_to_ela_delays.append(distance_matrix[a1][a2])
            num_min_ack = num_atomix//2 # the limited number of Atomix follower nodes to ack the leader Atomix nodes
            a_to_ela_delays = np.sort(a_to_ela_delays) # sort delay from lowest to highest
            avr_min_a_to_ela_delays.append(np.mean(a_to_ela_delays[:num_min_ack])) # only need first num_atomix//2 delays
        min_avr_atomix_atomix_delay = min(avr_min_a_to_ela_delays) # keep minimum delay

    # calculate fst
    FTSs = []
    for source in range(num_nodes):
        for destination in range(num_nodes):
            if source == destination:
                continue
            delay = 0
            is_controlled_by_single_controller = True
            is_controlled_by_same_middle_controller = True
            added_middle_highest = False
            for s in shortest_paths[source][destination]:
                # switch-controller delay
                delay += distance_matrix[s][controller_of[s]] * 4

                # controller-atomix delay
                # remove the c-to-a delay of source switch
                if len(average_atomix_delay_from) > 1:
                    flt_dca_css = {k: v for k,v in average_atomix_delay_from.items() if k != controller_of[source]}
                else:
                    flt_dca_css = average_atomix_delay_from
                c_of_dsmh = max(flt_dca_css, key=flt_dca_css.get) # highest c-to-a delay of swithes except source swith
                c_of_dsml = min(flt_dca_css, key=flt_dca_css.get) # lowest c-to-a delay of swithes except source swith
                flt_c_of_ss = [rest_c for rest_c in controller_of if rest_c != controller_of[source]] # remove controller of source switch from controller_of list
                is_controlled_by_same_middle_controller = len(set(flt_c_of_ss)) == 1 # check whether all switches except source switch controlled by same controller
                dss = average_atomix_delay_from[controller_of[source]] # c-to-a delay of source switch
                dsmh = flt_dca_css[c_of_dsmh] # c-to-a highest delay of middle switches
                dsml = flt_dca_css[c_of_dsml] # c-to-a lowest delay of middle switches
                dsd = average_atomix_delay_from[controller_of[destination]] # c-to-a delay of destination switch
                if s == source:
                    delay += dss * 2
                elif s != destination:
                    if controller_of[s] != controller_of[source]:
                        is_controlled_by_single_controller = False
                        if dsmh > dss and not added_middle_highest:
                            added_middle_highest = True
                            delay += dsmh
                else:
                    if controller_of[destination] == controller_of[source]:
                        if not is_controlled_by_single_controller:
                            delay += dss
                    else:
                        if is_controlled_by_same_middle_controller:
                            if dsd == dsmh:
                                delay += (dsd + dss)
                        else:
                            if dsd == dsmh:
                                delay += (dsd * 2 + dss)
                            if dsd == dsml:
                                delay += dsd
            
            # atomix-atomix delay
            delay +=  min_avr_atomix_atomix_delay * 3
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
    algorithm = AGEMOEA(pop_size=pop_size,
                      sampling=BinaryRandomSampling(),
                      crossover=TwoPointCrossover(),
                      mutation=BitflipMutation(),
                      eliminate_duplicates=True
                      )
    
    start = time.time()
    res = minimize(problem,
               algorithm,
               ('n_gen', 1000),
               seed=1,
               save_history=True,
               verbose=True)
    end = time.time()
    dt = end - start
    print(f'AGEMOEA - Elapsed time: {dt}')
    print('Total number of nodes: ', num_nodes)

    return res

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='Prefix for the result file')
    return parser.parse_args()

def main():
    args = parse_arg()

    res = optimize(simple_graph.graph)
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
    # res = optimize(Palmetto_graph.graph)

    F = res.F
    print(F[np.argsort(F[:, 0])])

    outputfile = 'res_bc_simple_agemoea.pkl'
    # outputfile = 'res_bc_Cogent_agemoea.pkl'
    # outputfile = 'res_bc_UsCarrier_agemoea.pkl'
    # outputfile = 'res_bc_HiberniaGlobal_agemoea.pkl'
    # outputfile = 'res_bc_Colt_agemoea.pkl'
    # outputfile = 'res_bc_Funet_agemoea.pkl'
    # outputfile = 'res_bc_Abvt_agemoea.pkl'
    # outputfile = 'res_bc_Intellifiber_agemoea.pkl'
    # outputfile = 'res_bc_DialtelecomCz_agemoea.pkl'
    # outputfile = 'res_bc_TataNld_agemoea.pkl'
    # outputfile = 'res_bc_Kdl_agemoea.pkl'
    # outputfile = 'res_bc_Internode_agemoea.pkl'
    # outputfile = 'res_bc_Missouri_agemoea.pkl'
    # outputfile = 'res_bc_Ion_agemoea.pkl'
    # outputfile = 'res_bc_Ntelos_agemoea.pkl'
    # outputfile = 'res_bc_Palmetto_agemoea.pkl'
    
    if(args.prefix):
        outputfile = args.prefix + outputfile
    with open(outputfile, 'wb') as f:
        pickle.dump(res,f)


if __name__ == "__main__":
    main()