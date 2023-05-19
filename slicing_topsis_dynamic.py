import random
import sys
import pickle
from datetime import datetime, date
import logging
import config
from network_attributes import NetworkAttribute
from entropy_slicing import WeightMatrix
import networkx as nx
from vne_p import create_vne
import math


class temp_map:
    def __init__(self, virtual_request, map=[]) -> None:
        self.node_map = map
        self.node_cost = 0
        self.node_cost += sum(virtual_request.node_weights.values())
        self.edge_cost = 0
        self.total_cost = sys.maxsize
        self.edge_map = dict()


def node_map(substrate, virtual, node_slice_list=None):
    embed_map = [0 for x in range(virtual.nodes)]
    # sn_rank = get_ranks(substrate)


    sn = dict()
    for i in range(substrate.nodes):
        ls = []
        for nd in substrate.neighbours[i]:
            ls.append(int(nd))
        sn[i] = ls
    
    sn_link_bw = {}
    _node_obj = NetworkAttribute(sn, crb=substrate.node_weights, link_bandwidth=sn_link_bw)
    sn_node_bw = _node_obj.normalized_node_bandwidth(substrate)
    sn_node_crb = _node_obj.normalized_crb(substrate)
    sn_crb = _node_obj.get_network_crb()
    sn_link_bw = _node_obj.get_link_bandwidth()
    
    sn_node_degree=_node_obj.normalized_node_degree()
    sn_node_security = _node_obj.normalized_security_probability(substrate)
    
    sn_btw_cnt = nx.betweenness_centrality(nx.DiGraph(sn))      #ADDED - inbuilt function for betweeness centrality
    sn_eigned_vct = nx.eigenvector_centrality(nx.DiGraph(sn), max_iter = 10000)   #ADDED - inbuilt function for eigenvector centrality
  
    sn_rank = WeightMatrix(sn, sn_node_crb, sn_node_bw, sn_btw_cnt,sn_eigned_vct,sn_node_degree,sn_node_security).compute_entropy_measure_matrix()

    _vne = dict()
    for i in range(virtual.nodes):
        ls = []
        for nd in virtual.neighbours[i]:
            ls.append(int(nd))
        _vne[i] = ls

    vne_node_attrib = NetworkAttribute(_vne, virtual=True)
    node_norm_bw = vne_node_attrib.normalized_node_bandwidth(virtual)
    node_norm_crb = vne_node_attrib.normalized_crb(virtual)
    node_bw = vne_node_attrib.get_network_bandwidth()
    node_crb = vne_node_attrib.get_network_crb()
    link_bw = vne_node_attrib.get_link_bandwidth()
    node_degree = vne_node_attrib.normalized_node_degree()
    vne_node_security = vne_node_attrib.normalized_security_probability(virtual)
    _vnode = [_vne, node_bw, node_crb, node_norm_bw,
                            node_norm_crb, link_bw, node_degree, vne_node_security]
    btw_cnt = nx.betweenness_centrality(nx.DiGraph(_vnode[0]))
    eigned_vct = nx.eigenvector_centrality(nx.DiGraph(_vnode[0]), max_iter = 1000)
    # norm_crb, norm_bw, btw_cnt, eigen_cnt, node_degree, security
    node_rank = WeightMatrix(_vnode[0], _vnode[4], _vnode[3], btw_cnt,
                             eigned_vct, _vnode[6], _vnode[7]).compute_entropy_measure_matrix() # Compute Weight of the attributes

    # TOPSIS based ranking for virtual request
    # vn_rank = get_ranks_virtual(virtual)
    # vn_rank = get_ranks_substrate(virtual)
    v_order = sorted([a for a in range(virtual.nodes)], key = lambda x : node_rank[x])
    if node_slice_list is None:
        # Embedding without slicing
        s_order = sorted([a for a in range(substrate.nodes)], key = lambda x : sn_rank[x])
    else:
        # if slice nodes are less than virtual nodes, return None
        if len(node_slice_list) < virtual.nodes:
            return None
        # ranking just the nodes in the slice
        s_order = sorted(node_slice_list, key = lambda x : sn_rank[x])

    assigned_nodes = set()
    for vnode in v_order:
        # TOPSIS based ranking for substrate network
        for snode in s_order:
            if substrate.node_weights[snode] >= virtual.node_weights[vnode] and substrate.security_probability[snode] >= virtual.security_probability[vnode]  and snode not in assigned_nodes:
                embed_map[vnode] = snode
                substrate.node_weights[snode] -= virtual.node_weights[vnode]
                assigned_nodes.add(snode)
                break
            else:
                if snode in assigned_nodes:
                    continue

                if substrate.node_weights[snode] < virtual.node_weights[vnode] and substrate.security_probability[snode] < virtual.security_probability[vnode]:
                    reason = "security and CRB requirements."
                elif substrate.security_probability[snode] < virtual.security_probability[vnode]:
                    reason = "security requirements."
                elif substrate.node_weights[snode] < virtual.node_weights[vnode]:
                    reason = "CRB requirements."

                logging.info(f"Dynamic: {vnode} mapping failed on {snode} due to {reason}")
            if snode == s_order[-1]:
                return None
    return embed_map
    


def edge_map(substrate, virtual, req_no, req_map,edge_slice=None):
    substrate_copy = pickle.loads(pickle.dumps(substrate,-1))
    if edge_slice is not None : 
        if len(edge_slice) == 0 :
            return False
        substrate_copy.edge_weights = pickle.loads(pickle.dumps(edge_slice,-1))
        substrate_copy.update_edges_and_neighbours(list(edge_slice.keys()))

    for edge in virtual.edges:
        if int(edge[0]) < int(edge[1]):
            weight = virtual.edge_weights[edge]
            link_security = virtual.link_security[edge]
            left_node = req_map.node_map[int(edge[0])]
            right_node = req_map.node_map[int(edge[1])]
            path = substrate_copy.findShortestPathWithLinkSecurity(str(left_node), str(right_node), weight, link_security)
            # path = substrate_copy.findShortestPath(str(left_node), str(right_node), weight)
            if len(path) > 0:
                req_map.edge_map[req_no, edge] = path
                for j in range(1, len(path)):
                    substrate_copy.edge_weights[(path[j - 1], path[j])] -= weight
                    substrate_copy.edge_weights[(path[j], path[j - 1])] -= weight
                    req_map.edge_cost += weight
            else:
                logging.info(f"\t\tLength of path is {len(path)}")
                return False

    ############### Info dumping substrate before mapping ###############
    sub_wt = []
    # sn_rank = get_ranks_substrate(substrate)
    sorder = sorted([a for a in range(substrate.nodes)])
    for node in sorder:
        sub_wt.append((node, substrate.node_weights[node], substrate.security_probability[node]))
    logging.info(f"\t\tSubstrate node before mapping VNR-{req_no} is {sub_wt}")
    sub_wt = []
    for edge in substrate.edges:
        sub_wt.append((edge, substrate.edge_weights[edge]))
    logging.info(f"\t\tSubstrate edge before mapping VNR-{req_no} is {sub_wt}")
    logging.info(f"\t\tNode map of VNR-{req_no} is {req_map.node_map}")
    logging.info(f"\t\tEdge map of VNR-{req_no} is {req_map.edge_map}")

    ############### Updating the subtrate resources post successful mapping ###############
    for edge, path in req_map.edge_map.items():
        edge = edge[1]
        for i in range(1, len(path)):
            substrate.edge_weights[(path[i - 1], path[i])] -= virtual.edge_weights[edge]
            substrate.edge_weights[(path[i], path[i-1])] -= virtual.edge_weights[edge]
    for node in range(virtual.nodes):
        substrate.node_weights[req_map.node_map[node]] -= virtual.node_weights[node]
        # substrate.rsa_embedding[req_map.node_map[node]] =virtual.rsa_values[node]
    sub_wt = []
    # sn_rank = get_ranks_substrate(substrate)
    sorder = sorted([a for a in range(substrate.nodes)])
    for node in sorder:
        sub_wt.append((node, substrate.node_weights[node], substrate.security_probability[node]))
    logging.info(f"\t\tSubstrate after mapping VNR-{req_no} is {sub_wt}")
    sub_wt = []
    for edge in substrate.edges:
        sub_wt.append((edge, substrate.edge_weights[edge]))
    logging.info(f"\t\tSubstrate edge after mapping VNR-{req_no} is {sub_wt}")
    return True

def findAvgPathLength(vnr):
    cnt=0
    for node1 in range(vnr.nodes):
        for node2 in range(vnr.nodes):
            if(node1 != node2):
                path = vnr.findShortestPath(str(node1), str(node2), 0)
                cnt += len(path)-1
    total_nodes = vnr.nodes
    cnt /= (total_nodes)*(total_nodes-1)
    return cnt

def generate_substrate_slice(substrate, bandwidth_threshold,slice_type) :
    high_slice_node_list = set()
    low_slice_node_list = set()
    high_edge_slice = dict()
    low_edge_slice = dict()

    # Selecting nodes
    for nodes, bandwidth in substrate.edge_weights.items():
        (a,b) = nodes
        if bandwidth > bandwidth_threshold :
            high_slice_node_list.add(int(a))
            high_slice_node_list.add(int(b))
    
    low_slice_node_list = [node for node in range(substrate.nodes) if node not in high_slice_node_list]
    # Selecting edge
    for nodes,bandwidth in substrate.edge_weights.items():
        (a,b) = nodes 
        if bandwidth > bandwidth_threshold :
            high_edge_slice[nodes] = bandwidth
        else :
            low_edge_slice[nodes] = bandwidth        
    if slice_type == "high" : 
        slice_node_list = high_slice_node_list
        edge_slice = high_edge_slice
    else : 
        slice_node_list = low_slice_node_list
        edge_slice = low_edge_slice

    return slice_node_list,edge_slice

def get_revenue(vne):
    cpu_sum = 0
    bandwidth_sum = 0

    for node in vne.node_weights:
        cpu_sum += vne.node_weights[node]

    for edge in vne.edge_weights:
        bandwidth_sum += vne.edge_weights[edge]

    revenue = cpu_sum + bandwidth_sum // 2
    return revenue


def slicing_topsis_dynamic():
    print(f"\t\t{datetime.now().time()}\tSlicing-Topsis-Dynamic Started")

    # reading substrate and vne from the config files
    substrate, vne_list = pickle.loads(pickle.dumps(config.substrate,-1)), pickle.loads(pickle.dumps(config.vne_list,-1))
    
    # copy of the substrate
    copy_sub = pickle.loads(pickle.dumps(substrate,-1))
    logging.basicConfig(filename="slicing-topsis-dynamic.log", filemode="w", level=logging.INFO)

    ############### Info dumping the initial substrate in the log file ############### 
    logging.info(f"\n\n\t\t\t\t\t\tSUBSTRATE NETWORK (BEFORE MAPPING VNRs)")
    logging.info(f"\t\tTotal number of nodes and edges in substrate network is : {substrate.nodes} and {len(substrate.edges)} ")
    temp = []
    for node in range(substrate.nodes):
        temp.append((node, substrate.node_weights[node]))
    logging.info(f"\t\tNodes of the substrate network with weight are : {temp}")
    temp = []
    for edge in substrate.edges:
        temp.append((edge, substrate.edge_weights[edge]))
    logging.info(f"\t\tEdges of the substrate network with weight are : {temp}\n\n\t\t\t\t\t\tVIRTUAL NETWORK")

    ############### Info dumping the number of VNRs in the log file ###############
    total_vnr_nodes = 0
    total_vnr_links = 0
    logging.info(f"\t\tTotal number of Virtual Network Request is : {len(vne_list)}\n")
    for vnr in range(len(vne_list)):
        logging.info(f"\t\tTotal number of nodes and edges in VNR-{vnr} is : {vne_list[vnr].nodes} and {len(vne_list[vnr].edges)}")
        temp = []
        total_vnr_nodes += vne_list[vnr].nodes 
        for node in range(vne_list[vnr].nodes):
            temp.append((node, vne_list[vnr].node_weights[node],vne_list[vnr].security_probability[node]))
        logging.info(f"\t\tNodes of the VNR-{vnr} with weight and security requirement(security Probability) are : {temp}")
        temp = []
        total_vnr_links += len(vne_list[vnr].edges) 
        for edge in vne_list[vnr].edges:
            temp.append((edge, vne_list[vnr].edge_weights[edge]))
        if vnr == len(vne_list)-1:
            logging.info(f"\t\tEdges of the VNR-{vnr} with weight are : {temp}\n\n")
        else:
            logging.info(f"\t\tEdges of the VNR-{vnr} with weight are : {temp}")        

    ############### Initializing the metrics ###############    
    start_time = datetime.now().time()
    slicing_accepted = 0
    remapping_accepted = 0 
    dynamic_accepted = 0
    revenue = 0
    path_cnt=0
    avg_path_length = 0
    curr_map = dict()  # only contains the requests which are successfully mapped
    pre_resource_edgecost = sum(substrate.edge_weights.values())//2 # total available bandwidth of the physical network
    pre_resource_nodecost = sum(substrate.node_weights.values()) # total crb bandwidth of the physical network
    pre_resource = pre_resource_edgecost + pre_resource_nodecost    

    requests_to_be_remapped = dict() # stores failed embeddings 

    vne_list = sorted(vne_list, key=lambda x: get_revenue(x))  # ascending order
    config.vne_list = pickle.loads(pickle.dumps(vne_list,-1)) # updates in the ordering of vnrs in the global vne_list
    req_order = list(range(len(vne_list))) # handles the ordering of the requests
    requests_to_be_remapped = dict() #stores if any of the requests need remapping after slicing based embedding
    
    for req_no in req_order:
        average_bandwidth_strength = sum(substrate.edge_weights.values())/len(substrate.edge_weights)
        # Categorizing a request as high bandwidth and low bandwidth as per priority
        if vne_list[req_no].vnr_priority :
            slice_node_list, edge_slice = generate_substrate_slice(substrate,average_bandwidth_strength,"high")
        else :
            slice_node_list,edge_slice = generate_substrate_slice(substrate,average_bandwidth_strength,"low")
        req_map = node_map(pickle.loads(pickle.dumps(substrate,-1)), vne_list[req_no], slice_node_list)
        if req_map is None:
            logging.info(f"Slicing : Node mapping not possible")
            logging.warning(f"\tNode mapping not possible for req no {req_no}\n")
            # Embedding failed, adding request for non-slicing based embedding
            requests_to_be_remapped[req_no] = vne_list[req_no]
            continue
        logging.info(f"Slicing : Node mapping success for req no {req_no}")       
        req_map = temp_map(vne_list[req_no], req_map)
        if not edge_map(substrate, vne_list[req_no], req_no, req_map,edge_slice):
            logging.info(f"Slicing : Edge mapping not possible for req no {req_no}")
            logging.warning(f"\tEdge mapping not possible for req no {req_no}\n")
            # Embedding failed, adding request for non-slicing based embedding
            requests_to_be_remapped[req_no] = vne_list[req_no]
            continue
        slicing_accepted += 1
        logging.info(f"Slicing : Edge mapping success for req no {req_no}")       
        avg_path_length += findAvgPathLength(vne_list[req_no])
        for edge in req_map.edge_map:
            path_cnt += len(req_map.edge_map[edge])
        req_map.total_cost = req_map.node_cost + req_map.edge_cost
        
        logging.info(f"\t\tMapping for request {req_no} is done successfully!! {req_map.node_map} with revenue {sum(vne_list[req_no].node_weights.values()) + sum(vne_list[req_no].edge_weights.values())//2} and total cost {req_map.total_cost}\n")
        curr_map[req_no] = req_map
        revenue += sum(vne_list[req_no].node_weights.values()) + sum(vne_list[req_no].edge_weights.values())//2
    
    
    logging.info(f"Number of requests that need remapping : {len(requests_to_be_remapped)}")
    logging.info(f"\t\tRemapping for f{len(requests_to_be_remapped)} requests\n\n")
    # Remapping post slicing based embeddding
    for req_no, virtual_request in requests_to_be_remapped.items() :
        req_map = node_map(pickle.loads(pickle.dumps(substrate,-1)),virtual_request)
        if req_map is None:
            logging.info(f"Remapping : Node mapping not possible for req no {req_no}")
            logging.warning(f"\tNode mapping not possible for req no {req_no}\n")
            continue
        logging.info(f"Remapping : Node mapping success for req no {req_no}")
        req_map = temp_map(virtual_request,req_map)
        if not edge_map(substrate, virtual_request, req_no, req_map):
            logging.info(f"Remapping : Edge mapping not possible for req no {req_no}")
            logging.warning(f"\tEdge mapping not possible for req no {req_no}\n")
            continue
        logging.info(f"Remapping : Edge mapping success for req no {req_no}")
        logging.info(f'\nRequest {req_no} remapped!\n')
        remapping_accepted += 1
        avg_path_length += findAvgPathLength(virtual_request)
        for edge in req_map.edge_map:
            path_cnt += len(req_map.edge_map[edge])
        req_map.total_cost = req_map.node_cost + req_map.edge_cost
        curr_map[req_no] = req_map
        revenue += sum(vne_list[req_no].node_weights.values()) + sum(vne_list[req_no].edge_weights.values())//2
        logging.info(f"\t\tMapping for request {req_no} is done successfully!! {req_map.node_map} with revenue {sum(vne_list[req_no].node_weights.values()) + sum(vne_list[req_no].edge_weights.values())//2} and total cost {req_map.total_cost}\n")
    
    # 1. Select 25% of VNRs at random and update the VNRs randomly
    # 2. Check if the current mapping is sufficient to accomodate the updated VNEs
    # 3. If sufficient, continue to the next updated VNE
    # 4. If insufficient, release the VMs that cannot be accomodated and re-embed them over the substrate
    # 5. Perform link mapping
    # 6. Update resources
    
    num_of_dynamic_vnrs = math.ceil(0.25*len(vne_list))
    # Indices of VNEs to be changed in the VNE list
    vnrs_for_dynamic = random.sample(req_order,num_of_dynamic_vnrs)
    logging.info(f"Dynamic : VNRs to be re-embedded : {vnrs_for_dynamic}")
    # check if the selected VNEs were succesfully embedded, if so release resources
    for vnr_number in vnrs_for_dynamic :
        if vnr_number in curr_map :
            mapping_for_the_vne = curr_map[vnr_number]
            virtual = vne_list[vnr_number]
            # release CRB to servers
            for node in range(virtual.nodes):
                substrate.node_weights[mapping_for_the_vne.node_map[node]] += virtual.node_weights[node]           
            # Release bandwidth 
            for edge, path in mapping_for_the_vne.edge_map.items():
                edge = edge[1]
                for i in range(1, len(path)):
                    substrate.edge_weights[(path[i - 1], path[i])] += virtual.edge_weights[edge]
                    substrate.edge_weights[(path[i], path[i-1])] += virtual.edge_weights[edge]
            # Deduct from revenue
            avg_path_length -= findAvgPathLength(virtual)
            revenue -= sum(virtual.node_weights.values()) + (sum(virtual.edge_weights.values())//2)
            del curr_map[vnr_number]
            # Decrement from accepted or remapped
            if vnr_number in requests_to_be_remapped :
                remapping_accepted -= 1
            else :
                slicing_accepted -= 1
    # Generate replacement VNEs
    additional_vne = []
    for vne_number in vnrs_for_dynamic : 
        no_of_nodes = vne_list[vne_number].nodes
        new_vne = create_vne(min_nodes=no_of_nodes, max_nodes=no_of_nodes, no_requests=1)
        additional_vne.append(new_vne[0])
    # TODO : improve
    i = 0
    for vnr_number in vnrs_for_dynamic : 
        vne_list[vnr_number] = additional_vne[i]
        i += 1
    
    for req_no in vnrs_for_dynamic :
        virtual_request = vne_list[req_no]
        req_map = node_map(pickle.loads(pickle.dumps(substrate,-1)),virtual_request)
        if req_map is None:
            logging.info(f"Dynamic Re-embedding : Node mapping not possible for req no {req_no}")
            logging.warning(f"\tNode mapping not possible for req no {req_no}\n")
            continue
        logging.info(f"Dynamic Re-embedding : Node mapping success for req no {req_no}")
        req_map = temp_map(virtual_request,req_map)
        if not edge_map(substrate, virtual_request, req_no, req_map):
            logging.info(f"Dynamic Re-embedding : Edge mapping not possible for req no {req_no}")
            logging.warning(f"\tEdge mapping not possible for req no {req_no}\n")
            continue
        logging.info(f"Dynamic Re-embedding : Edge mapping success for req no {req_no}")
        logging.info(f'\nRequest {req_no} remapped!\n')
        dynamic_accepted += 1
        avg_path_length += findAvgPathLength(virtual_request)
        for edge in req_map.edge_map:
            path_cnt += len(req_map.edge_map[edge])
        req_map.total_cost = req_map.node_cost + req_map.edge_cost
        curr_map[req_no] = req_map
        revenue += sum(vne_list[req_no].node_weights.values()) + (sum(vne_list[req_no].edge_weights.values())//2)
        logging.info(f"\t\tMapping for request {req_no} is done successfully!! {req_map.node_map} with revenue {sum(vne_list[req_no].node_weights.values()) + sum(vne_list[req_no].edge_weights.values())//2} and total cost {req_map.total_cost}\n")

    config.curr_map = pickle.loads(pickle.dumps(curr_map,-1)) # saving the current mapping to be used during re-embedding

    ### Info dumping substrate network after embedding ###
    logging.info(f"\n\n\t\t\t\t\t\tSUBSTRATE NETWORK AFTER MAPPING VNRs")
    logging.info(f"\t\tTotal number of nodes and edges in substrate network is : {substrate.nodes} and {len(substrate.edges)} ")
    temp = []
    for node in range(substrate.nodes):
        temp.append((node, substrate.node_weights[node], substrate.security_probability[node]))
    logging.info(f"\t\tNodes of the substrate network with weight and attack probability are : {temp}")
    temp = []
    for edge in substrate.edges:
        temp.append((edge, substrate.edge_weights[edge]))
    logging.info(f"\t\tEdges of the substrate network with weight are : {temp}\n\n")   

    ### Metrics calculation ###
    ed_cost  = 0
    no_cost = 0
    for request in curr_map.values():
        ed_cost += request.edge_cost # total bandwidth for all the mapped requests
        no_cost += request.node_cost # total crb for all the mapped requests

    tot_cost = ed_cost + no_cost

    post_resource_edgecost =0
    post_resource_nodecost=0
    utilized_nodes=0
    utilized_links=0
    utilized_servers = set()
    average_node_utilization = 0
    average_edge_utilization = 0
    for edge in substrate.edge_weights:
        post_resource_edgecost += substrate.edge_weights[edge]
        if substrate.edge_weights[edge] != copy_sub.edge_weights[edge]:
            utilized_links += 1
            average_edge_utilization += (
                    (copy_sub.edge_weights[edge] - substrate.edge_weights[edge]) / copy_sub.edge_weights[edge])
            logging.info(
                f"The edge utilization of substrate edge {edge} is {((copy_sub.edge_weights[edge] - substrate.edge_weights[edge]) / copy_sub.edge_weights[edge]) * 100:0.4f}")
    post_resource_edgecost //= 2
    if utilized_links != 0:
        average_edge_utilization = average_edge_utilization / 2
        average_edge_utilization /= (utilized_links // 2)
    for node in substrate.node_weights:
        post_resource_nodecost += substrate.node_weights[node]
        if substrate.node_weights[node] != copy_sub.node_weights[node]:
            utilized_nodes += 1
            utilized_servers.add(node)
            average_node_utilization += (
                    (copy_sub.node_weights[node] - substrate.node_weights[node]) / copy_sub.node_weights[node])
            logging.info(
                f"The node utilization of the substrate node:{node} is {((copy_sub.node_weights[node] - substrate.node_weights[node]) / copy_sub.node_weights[node]) * 100:0.4f}")
    if utilized_nodes != 0:
        average_node_utilization /= utilized_nodes

    total_accepted = slicing_accepted + remapping_accepted + dynamic_accepted
    if total_accepted != 0:
        avg_path_length /= total_accepted
    
    post_resource = post_resource_edgecost + post_resource_nodecost

    end_time = datetime.now().time()
    duration = datetime.combine(date.min, end_time) - datetime.combine(date.min, start_time)
    # logging.info("Start time",start_time)
    # logging.info("End time", end_time)
    # logging.info("Duration",duration)
    substrate.successful_requests = total_accepted
    config.re_embed_substrate = pickle.loads(pickle.dumps(substrate,-1)) # copy the current substrate state to be used in re-embedding stage
    config.re_embed_vne_list = pickle.loads(pickle.dumps(vne_list,-1)) # saves the order of the VNRs
    config.active_servers = pickle.loads(pickle.dumps(utilized_servers,-1)) # saves the servers used up after embedding
    logging.info(f"\t\tThe revenue is {revenue} and total cost is {tot_cost}")
    if tot_cost == 0:
        logging.error(f"\t\tCouldn't embed any request")
        output_dict = {
            "revenue": -1,
            "total_cost": -1,
            "slicing_accepted": -1,
            "remapping_accepted" : -1,
            "dynamic_accepted" : -1,
            "accepted" : -1,
            "total_request": -1,
            "pre_resource": -1,
            "post_resource": -1,
            "avg_bw": -1,
            "avg_crb": -1,
            "avg_link": -1,
            "avg_node": -1,
            "avg_path": -1,
            "avg_exec": (duration),
            "total_nodes": total_vnr_nodes,
            "total_links": total_vnr_links,
            "No_of_Links_used" : -1,
            "No_of_Nodes_used" : -1
        }
        print(f"\t\t{datetime.now().time()}\tSlicing-Topsis completed\n")
        return output_dict
    logging.info(f"\t\tThe revenue is {revenue} and total cost is {tot_cost}")
    logging.info(f"\t\tThe revenue to cost ratio is {(revenue/tot_cost)*100:.4f}%")
    logging.info(f"\t\tTotal number of requests embedded is {total_accepted} out of {len(vne_list)}")
    logging.info(f"\t\tEmbedding ratio is {(total_accepted/len(vne_list))*100:.4f}%\n")
    logging.info(f"\t\tTotal {utilized_nodes} nodes are utilized out of {len(substrate.node_weights)}")
    logging.info(f"\t\tTotal {utilized_links//2} links are utilized out of {len(substrate.edge_weights)//2}\n")
    logging.info(f"\t\tAverage node utilization is {(utilized_nodes/len(substrate.node_weights))*100:0.4f}")
    logging.info(f"\t\tAverage link utilization is {(utilized_links/len(substrate.edge_weights))*100:0.4f}\n")
    logging.info(f"\t\tAvailabe substrate before embedding CRB: {pre_resource_nodecost} BW: {pre_resource_edgecost} total: {pre_resource}")
    logging.info(f"\t\tAvailabe substrate after embedding CRB: {post_resource_nodecost} BW: {post_resource_edgecost} total: {post_resource}")
    logging.info(f"\t\tConsumed substrate CRB: {pre_resource_nodecost-post_resource_nodecost} BW: {pre_resource_edgecost-post_resource_edgecost} total: {pre_resource - post_resource}\n")
    logging.info(f"\t\tAverage Path length is {avg_path_length:.4f}\n")
    logging.info(f"\t\tAverage BW utilization {(ed_cost/pre_resource_edgecost)*100:.4f}%")
    logging.info(f"\t\tAverage CRB utilization {(no_cost/pre_resource_nodecost)*100:.4f}%")
    logging.info(f"\t\tAverage execution time {duration/len(vne_list)} (HH:MM:SS)")
    logging.shutdown()
    output_dict = {
        "revenue": revenue,
        "total_cost": tot_cost,
        "slicing_accepted" : slicing_accepted,
        "remapping_accepted" : remapping_accepted,
        "dynamic_accepted" : dynamic_accepted,
        "accepted" : total_accepted,
        "total_request": len(vne_list),
        "pre_resource": pre_resource,
        "post_resource": post_resource,
        "avg_bw": (average_edge_utilization) * 100,
        "avg_crb": (average_node_utilization) * 100,
        "avg_link": ((utilized_links / len(substrate.edge_weights)) ) * 100, #2
        "No_of_Links_used": (utilized_links // 2),
        "avg_node": (utilized_nodes / len(substrate.node_weights)) * 100,
        "No_of_Nodes_used": (utilized_nodes),
        "avg_path": avg_path_length,
        "avg_exec": (duration),
        "total_nodes": total_vnr_nodes,
        "total_links": total_vnr_links // 2,
    }
    print(f"\t\t{datetime.now().time()}\tSlicing-Topsis-Dynamic completed\n")
    return output_dict


if __name__ == '__main__':
    '''
    The output can be different as every time we shuffle the
    request order of VNRs.
    '''

    slicing_topsis_dynamic()
