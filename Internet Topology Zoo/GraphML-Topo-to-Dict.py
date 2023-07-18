#!/usr/bin/python

#GraphML-Topo-to-Mininet-Network-Generator
#
# This file parses Network Topologies in GraphML format from the Internet Topology Zoo.
# A python file for creating Mininet Topologies will be created as Output.
# Files have to be in the same directory.
#
# Arguments:
#   -f              [filename of GraphML input file]
#   --file          [filename of GraphML input file]
#   -o              [filename of GraphML output file]
#   --output        [filename of GraphML output file]
#   -b              [number as integer for bandwidth in mbit]
#   --bw            [number as integer for bandwidth in mbit]
#   --bandwidth     [number as integer for bandwidth in mbit]
#   -c              [controller ip as string]
#   --controller    [controller ip as string]
#
# Without any input, program will terminate.
# Without specified output, outputfile will have the same name as the input file.
# This means, the argument for the outputfile can be omitted.
# Parameters for bandwith and controller ip have default values, if they are omitted, too.
#
#
# sjas
# Wed Jul 17 02:59:06 PDT 2013
#
#
# TODO's:
#   -   fix double name error of some topologies
#   -   fix topoparsing (choose by name, not element <d..>)
#           =    topos with duplicate labels
#   -   use 'argparse' for script parameters, eases help creation
#
#################################################################################



import xml.etree.ElementTree as ET
import sys
import math
import re
from sys import argv
import pprint

input_file_name = ''
output_file_name = ''
bandwidth_argument = ''
controller_ip = ''

# first check commandline arguments
for i in range(len(argv)):

    if argv[i] == '-f':
        input_file_name = argv[i+1]
    if argv[i] == '--file':
        input_file_name = argv[i+1]
    if argv[i] == '-o':
        output_file_name = argv[i+1]
    if argv[i] == '--output':
        output_file_name = argv[i+1]
    if argv[i] == '-b':
        bandwidth_argument = argv[i+1]
    if argv[i] == '--bw':
        bandwidth_argument = argv[i+1]
    if argv[i] == '--bandwidth':
        bandwidth_argument = argv[i+1]
    if argv[i] == '-c':
        controller_ip = argv[i+1]
    if argv[i] == '--controller':
        controller_ip = argv[i+1]

# terminate when inputfile is missing
if input_file_name == '':
    sys.exit('\n\tNo input file was specified as argument....!')

# define string fragments for output later on
outputstring_1 = '''#!/usr/bin/python

"""
Topology Dict, generated by GraphML-Topo-to-Dict.
"""
graph = {
'''

#WHERE TO PUT RESULTS
outputstring_to_be_exported = ''
outputstring_to_be_exported += outputstring_1

#READ FILE AND DO ALL THE ACTUAL PARSING IN THE NEXT PARTS
xml_tree    = ET.parse(input_file_name)
namespace   = "{http://graphml.graphdrawing.org/xmlns}"
ns          = namespace # just doing shortcutting, namespace is needed often.

#GET ALL ELEMENTS THAT ARE PARENTS OF ELEMENTS NEEDED LATER ON
root_element    = xml_tree.getroot()
graph_element   = root_element.find(ns + 'graph')

# GET ALL ELEMENT SETS NEEDED LATER ON
index_values_set    = root_element.findall(ns + 'key')
node_set            = graph_element.findall(ns + 'node')
edge_set            = graph_element.findall(ns + 'edge')

# SET SOME VARIABLES TO SAVE FOUND DATA FIRST
# memomorize the values' ids to search for in current topology
node_label_name_in_graphml = ''
node_latitude_name_in_graphml = ''
node_longitude_name_in_graphml = ''
# for saving the current values
node_index_value     = ''
node_name_value      = ''
node_longitude_value = ''
node_latitude_value  = ''
# id:value dictionaries
id_node_name_dict   = {}     # to hold all 'id: node_name_value' pairs
id_longitude_dict   = {}     # to hold all 'id: node_longitude_value' pairs
id_latitude_dict    = {}     # to hold all 'id: node_latitude_value' pairs

# FIND OUT WHAT KEYS ARE TO BE USED, SINCE THIS DIFFERS IN DIFFERENT GRAPHML TOPOLOGIES
for i in index_values_set:

    if i.attrib['attr.name'] == 'label' and i.attrib['for'] == 'node':
        node_label_name_in_graphml = i.attrib['id']
    if i.attrib['attr.name'] == 'Longitude':
        node_longitude_name_in_graphml = i.attrib['id']
    if i.attrib['attr.name'] == 'Latitude':
        node_latitude_name_in_graphml = i.attrib['id']

# NOW PARSE ELEMENT SETS TO GET THE DATA FOR THE TOPO
# GET NODE_NAME DATA
# GET LONGITUDE DATK
# GET LATITUDE DATA
for n in node_set:

    node_index_value = n.attrib['id']

    #get all data elements residing under all node elements
    data_set = n.findall(ns + 'data')

    #finally get all needed values
    for d in data_set:

        #node name
        if d.attrib['key'] == node_label_name_in_graphml:
            #strip all whitespace from names so they can be used as id's
            node_name_value = re.sub(r'\s+', '', d.text)
        #longitude data
        if d.attrib['key'] == node_longitude_name_in_graphml:
            node_longitude_value = d.text
        #latitude data
        if d.attrib['key'] == node_latitude_name_in_graphml:
            node_latitude_value = d.text

        #save id:data couple
        id_node_name_dict[node_index_value] = node_name_value
        id_longitude_dict[node_index_value] = node_longitude_value
        id_latitude_dict[node_index_value]  = node_latitude_value

# SECOND CALCULATE DISTANCES BETWEEN SWITCHES,
#   set global bandwidth and create the edges between switches,
#   and link each single host to its corresponding switch

graph = {}
distance = 0.0
latency = 0.0

for e in edge_set:

    # GET IDS FOR EASIER HANDLING
    src_id = e.attrib['source']
    dst_id = e.attrib['target']

    # CALCULATE DELAYS

    #    CALCULATION EXPLANATION
    #
    #    formula: (for distance)
    #    dist(SP,EP) = arccos{ sin(La[EP]) * sin(La[SP]) + cos(La[EP]) * cos(La[SP]) * cos(Lo[EP] - Lo[SP])} * r
    #    r = 6378.137 km
    #
    #    formula: (speed of light, not within a vacuumed box)
    #     v = 1.97 * 10**8 m/s
    #
    #    formula: (latency being calculated from distance and light speed)
    #    t = distance / speed of light
    #    t (in ms) = ( distance in km * 1000 (for meters) ) / ( speed of light / 1000 (for ms))

    #    ACTUAL CALCULATION: implementing this was no fun.

    first_product               = math.sin(float(id_latitude_dict[dst_id])) * math.sin(float(id_latitude_dict[src_id]))
    second_product_first_part   = math.cos(float(id_latitude_dict[dst_id])) * math.cos(float(id_latitude_dict[src_id]))
    second_product_second_part  = math.cos((float(id_longitude_dict[dst_id])) - (float(id_longitude_dict[src_id])))

    distance = math.radians(math.acos(first_product + (second_product_first_part * second_product_second_part))) * 6378.137

    # t (in ms) = ( distance in km * 1000 (for meters) ) / ( speed of light / 1000 (for ms))
    # t         = ( distance       * 1000              ) / ( 1.97 * 10**8   / 1000         )
    latency = ( distance * 10000 ) / ( 197000 )

    src_id_i = int(src_id)
    dst_id_i = int(dst_id)

    if(not src_id_i in graph):
        graph[src_id_i] = {}
    if(not dst_id_i in graph):
        graph[dst_id_i] = {}
    
    graph[src_id_i][dst_id_i] = latency
    graph[dst_id_i][src_id_i] = latency



for k in sorted(graph.keys()):
    outputstring_to_be_exported += f'    {k}: {graph[k]}\n'

outputstring_to_be_exported+='}\n'

# GENERATION FINISHED, WRITE STRING TO FILE
outputfile = ''
if output_file_name == '':
    output_file_name = input_file_name + '-Dict.py'

outputfile = open(output_file_name, 'w')
outputfile.write(outputstring_to_be_exported)
outputfile.close()

print("Topology generation SUCCESSFUL!")