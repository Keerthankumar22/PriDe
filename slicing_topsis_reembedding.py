import logging
import config 
import sys
import pickle
import random
from datetime import datetime, date
from slicing_topsis import findAvgPathLength
from network_attributes import NetworkAttribute
from entropy_slicing import WeightMatrix
import networkx as nx
class temp_map:
    def __init__(self, virtual_request, map=[]) -> None:
        self.node_map = map
        self.node_cost = 0
        self.node_cost += sum(virtual_request.node_weights.values())
        self.edge_cost = 0
        self.total_cost = sys.maxsize
        self.edge_map = dict()


def node_map(substrate, virtual,server_list, vm_list):

    embed_map = dict()
    
    # Topsis based ranking for node embedding
    # vn_rank = get_ranks_virtual(virtual)
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
    

    s_order = sorted(server_list,key = lambda x: sn_rank[x])
    v_order = sorted(vm_list,key = lambda x: node_rank[x])
    assigned_nodes = set()
    for vnode in v_order:
        for snode in s_order:
            if substrate.node_weights[snode] >= virtual.node_weights[vnode] and substrate.security_probability[snode] >= virtual.security_probability[vnode]  and snode not in assigned_nodes:
                embed_map[vnode] = snode
                substrate.node_weights[snode] -= virtual.node_weights[vnode]
                assigned_nodes.add(snode)
                break
            if snode in assigned_nodes:
                continue

            if substrate.node_weights[snode] < virtual.node_weights[vnode] and substrate.security_probability[snode] < virtual.security_probability[vnode]:
                reason = "security and CRB requirements."
            elif substrate.security_probability[snode] < virtual.security_probability[vnode]:
                reason = "security requirements."
            elif substrate.node_weights[snode] < virtual.node_weights[vnode]:
                reason = "CRB requirements."

            logging.info(f"Re-embedding: {vnode} mapping failed on {snode} due to {reason}")
            if snode == s_order[-1]:
                return None
    return embed_map
    
def edge_map(substrate, virtual, req_no, req_map):
    substrate_copy = pickle.loads(pickle.dumps(substrate,-1))

    for edge in virtual.edges:
        if int(edge[0]) < int(edge[1]):
            weight = virtual.edge_weights[edge]
            link_security = virtual.link_security[edge]
            left_node = req_map.node_map[int(edge[0])]
            right_node = req_map.node_map[int(edge[1])]
            path = substrate_copy.findShortestPathWithLinkSecurity(str(left_node), str(right_node), weight, link_security)
            if len(path) > 0:
                req_map.edge_map[req_no, edge] = path
                for j in range(1, len(path)):
                    substrate_copy.edge_weights[(path[j - 1], path[j])] -= weight
                    substrate_copy.edge_weights[(path[j], path[j - 1])] -= weight
                    req_map.edge_cost += weight
            else:
                return None
    return req_map


def release_vnr_from_substrate(substrate,virtual,curr_map) : 
    for vm,server in enumerate(curr_map.node_map) :
        substrate.node_weights[server] += virtual.node_weights[vm]
    
    for edge, path in curr_map.edge_map.items():
            edge = edge[1]
            for i in range(1, len(path)):
                substrate.edge_weights[(path[i - 1], path[i])] += virtual.edge_weights[edge]
                substrate.edge_weights[(path[i], path[i-1])] += virtual.edge_weights[edge]
    

    return substrate

def reembedding() :
    print(f"\t\t{datetime.now().time()}\tSlicing-Topsis Re-embedding Started")
    
    substrate = pickle.loads(pickle.dumps(config.re_embed_substrate,-1))
    copy_sub = pickle.loads(pickle.dumps(config.substrate,-1))
    vne_list = pickle.loads(pickle.dumps(config.re_embed_vne_list,-1))
    request_mappings = pickle.loads(pickle.dumps(config.curr_map,-1))
    active_servers = pickle.loads(pickle.dumps(config.active_servers,-1))
    previous_accepted = substrate.successful_requests

    logging.basicConfig(filename="slicing-reembedding.log", filemode="w", level=logging.INFO)
    # Select 2 to 3 servers to attack from active servers
    number_of_attacked_servers = random.randint(2,3)
    # If 2 or more servers are active, attack 2 to 3 at random
    if len(active_servers) > 2 :
        attacked_servers = random.sample(active_servers,number_of_attacked_servers)
    else :
        # Else select all servers
        attacked_servers = active_servers

    logging.info(f"Re-embedding : Attacked servers {attacked_servers}")
    safe_servers = list(set(range(substrate.nodes))-set(attacked_servers))
    logging.info(f"Re-embedding : Safe servers {safe_servers}")
    req_order = list(range(len(vne_list)))

    # Initializing metrics
    start_time = datetime.now().time()
    vnrs_attacked = 0
    vnrs_reembed = 0 
    revenue = 0
    avg_path_length = 0
    pre_resource_edgecost = sum(substrate.edge_weights.values())//2
    pre_resource_nodecost = sum(substrate.node_weights.values())
    pre_resource = pre_resource_edgecost + pre_resource_nodecost
    total_vnr_nodes = 0
    total_vnr_links = 0
    path_cnt = 0
    fault_tolerance_performance = 0
    final_reembed_mapping = request_mappings

    # Fetching revenue from previous module
    for req_no in request_mappings :
        virtual = vne_list[req_no]
        revenue += sum(vne_list[req_no].node_weights.values()) + (sum(vne_list[req_no].edge_weights.values())//2)
    # Iterate through all VNRs
    for req_no in req_order :
        # if the request was not embedded successfully, no mapping for it exists hence continue
        if req_no not in request_mappings : 
            continue
        curr_map = request_mappings[req_no]
        logging.info(f"Mapped servers for {req_no} : {curr_map.node_map}")
        virtual = vne_list[req_no] # the current vnr
        vms_to_be_embedded = list()
        # Iterate through all the VMs and check if they are mapped to any of the attacked servers
        for vm in range(virtual.nodes) :
            vm_mapping = curr_map.node_map[vm]
            # Add condition for RSA check
            rsa_for_vm = virtual.rsa_values[vm]
            if vm_mapping in attacked_servers : 
                # Append to the list of nodes to be embedded
                # RSA check
                
                if rsa_for_vm not in substrate.rsa_embedding :
                    vms_to_be_embedded.append(vm)
                
                # Release cpu resource
                substrate.node_weights[vm_mapping] += virtual.node_weights[vm]
        # If there are no VMs on attacked servers
        if len(vms_to_be_embedded) == 0 :
            logging.info(f"No re-embedding required for {req_no}")
            continue
        vnrs_attacked += 1
        available_servers = list(set(safe_servers) - set(request_mappings[req_no].node_map))
        # Perform node embedding as previously done
        reembedded_vm_mapping = node_map(pickle.loads(pickle.dumps(substrate,-1)),virtual,available_servers,vms_to_be_embedded)
        logging.info(f"Re-embedded vulnerable VMs: {reembedded_vm_mapping}")
        if reembedded_vm_mapping is None : 
            # Node mapping not possible
            logging.info(f"Re-embedding : Node mapping not possible not possible for req_no{req_no}")
            # Release VNR from Substrate
            substrate = release_vnr_from_substrate(pickle.loads(pickle.dumps(substrate,-1)),virtual)
            # Decrease revenue
            revenue -= sum(vne_list[req_no].node_weights.values()) + sum(vne_list[req_no].edge_weights.values())//2
            # Remove mapping
            del final_reembed_mapping[req_no]
            continue
        # Update the current mapping array 
        temp_mapping = pickle.loads(pickle.dumps(curr_map,-1))
        for vm in reembedded_vm_mapping :
            temp_mapping.node_map[vm] = reembedded_vm_mapping[vm]
        logging.info(f"Re-embedded mapping array : {temp_mapping.node_map}")
        req_map = temp_map(virtual,temp_mapping.node_map)
        
        # Release link resources
        for edge, path in curr_map.edge_map.items():
            edge = edge[1]
            for i in range(1, len(path)):
                substrate.edge_weights[(path[i - 1], path[i])] += virtual.edge_weights[edge]
                substrate.edge_weights[(path[i], path[i-1])] += virtual.edge_weights[edge]
        
        # Perform link mapping 
        req_map = edge_map(substrate,virtual,req_no,req_map) 
        if req_map is None :
            # Edge mapping failed
            logging.info(f"Re-embedding : Edge mapping unsuccesful for req_no{req_no}")
            substrate = release_vnr_from_substrate(pickle.loads(pickle.dumps(substrate,-1)),virtual,curr_map)
            revenue -= sum(vne_list[req_no].node_weights.values()) + sum(vne_list[req_no].edge_weights.values())//2
            del final_reembed_mapping[req_no]
            continue 
        logging.info(f"Link embedding for req_no {req_no} : {req_map.edge_map}")
        # Updating the substrate post successful embedding
        for edge, path in req_map.edge_map.items():
            edge = edge[1]
            for i in range(1, len(path)):
                substrate.edge_weights[(path[i - 1], path[i])] -= virtual.edge_weights[edge]
                substrate.edge_weights[(path[i], path[i-1])] -= virtual.edge_weights[edge]
        for node in range(virtual.nodes):
            if node in reembedded_vm_mapping :
                # For all newly embedded VMs, update the substrate
                substrate.node_weights[req_map.node_map[node]] -= virtual.node_weights[node]
        final_reembed_mapping[req_no] = req_map
        vnrs_reembed += 1
        avg_path_length+=findAvgPathLength(virtual)
        for edge in req_map.edge_map:
            path_cnt += len(req_map.edge_map[edge])
        req_map.total_cost = req_map.node_cost + req_map.edge_cost
        
        total_vnr_nodes += virtual.nodes
        total_vnr_links += len(virtual.edges) 

    # Calculate metrics 
    ed_cost  = 0
    no_cost = 0
    for request in final_reembed_mapping.values():
        ed_cost += request.edge_cost # total bandwidth for all the re-embedded requests
        no_cost += request.node_cost # total crb for all the re-embedded requests

    tot_cost = ed_cost + no_cost
    post_resource_edgecost =0
    post_resource_nodecost=0
    utilized_nodes=0
    utilized_links=0
    average_edge_utilization = 0
    average_node_utilization = 0
    for edge in substrate.edge_weights:
        post_resource_edgecost += substrate.edge_weights[edge]
        if substrate.edge_weights[edge] != copy_sub.edge_weights[edge]:
            utilized_links += 1
            average_edge_utilization += (
                    (copy_sub.edge_weights[edge] - substrate.edge_weights[edge]) / copy_sub.edge_weights[edge])
            if copy_sub.edge_weights[edge] : 
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
            average_node_utilization += (
                    (copy_sub.node_weights[node] - substrate.node_weights[node]) / copy_sub.node_weights[node])
            if copy_sub.node_weights[node] > 0 :
                logging.info(
                f"The node utilization of the substrate node:{node} is {((copy_sub.node_weights[node] - substrate.node_weights[node]) / copy_sub.node_weights[node]) * 100:0.4f}")
    if utilized_nodes != 0:
        average_node_utilization /= utilized_nodes

    if vnrs_reembed != 0:
        avg_path_length /= vnrs_reembed
    
    total_accepted = previous_accepted + (vnrs_reembed - vnrs_attacked)
    fault_tolerance_performance = (total_accepted + vnrs_reembed) / len(vne_list) 
    post_resource = post_resource_edgecost + post_resource_nodecost
        
    logging.info(f"Total attacked VNRs : {vnrs_attacked}")
    logging.info(f"Total successfully re-embedded VNRs : {vnrs_reembed}")

    end_time = datetime.now().time()
    duration = datetime.combine(date.min, end_time) - datetime.combine(date.min, start_time)
    if vnrs_attacked == 0 or tot_cost == 0 :
        output_dict = {
            "revenue": -1,
            "total_cost": -1,
            "accepted": -1,
            "total_request": -1,
            "pre_resource": -1,
            "post_resource": -1,
            "avg_bw": -1,
            "avg_crb": -1,
            "avg_link": -1,
            "No_of_Links_used": -1,
            "avg_node": -1,
            "No_of_Nodes_used": -1,
            "avg_path": -1,
            "avg_exec": duration,
            "total_nodes": -1,
            "total_links": -1,
            "fault_tolerance_performance" : -1,
            "attacked" : -1,
            "recovered" : -1,
            "recovery_ratio" : -1,
        }
        print(f"\t\t{datetime.now().time()}\tSlicing-Topsis Re-embedding completed\n")
        return output_dict

    output_dict = {
            "revenue": revenue,
            "total_cost": tot_cost,
            "accepted": total_accepted,
            "total_request": len(vne_list),
            "attacked" : vnrs_attacked,
            "recovered" : vnrs_reembed,
            "recovery_ratio" : (vnrs_reembed/vnrs_attacked)*100,
            "pre_resource": pre_resource,
            "post_resource": post_resource,
            "avg_bw": (average_edge_utilization) * 100,
            "avg_crb": (average_node_utilization) * 100,
            "avg_link": ((utilized_links / len(substrate.edge_weights)) ) * 100, #/ 2
            "No_of_Links_used": (utilized_links // 2),
            "avg_node": (utilized_nodes / len(substrate.node_weights)) * 100,
            "No_of_Nodes_used": (utilized_nodes),
            "avg_path": avg_path_length,
            "avg_exec": duration,
            "total_nodes": total_vnr_nodes,
            "total_links": total_vnr_links,
            "fault_tolerance_performance" : fault_tolerance_performance
    }
    print(f"\t\t{datetime.now().time()}\tSlicing-Topsis Re-embedding completed\n")
    return output_dict

if __name__ == '__main__':
    reembedding()
