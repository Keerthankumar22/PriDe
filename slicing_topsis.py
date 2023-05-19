import sys
import pickle
from datetime import datetime, date
import logging
import config
# from slicing_topsis_substrate_helper import get_ranks as get_ranks_substrate
# from slicing_topsis_virtual_helper import get_ranks as get_ranks_virtual


class temp_map:
    def __init__(self, virtual_request, map=[]) -> None:
        self.node_map = map
        self.node_cost = 0
        self.node_cost += sum(virtual_request.node_weights.values())
        self.edge_cost = 0
        self.total_cost = sys.maxsize
        self.edge_map = dict()


def node_map(substrate, virtual, req_no, node_slice_list=None):
    embed_map = [0 for x in range(virtual.nodes)]

    # TOPSIS based ranking for virtual request
    # v_order = get_ranks_virtual(virtual) # Limited attributes
    v_order = get_ranks_substrate(virtual)  # with same set of attributes
    assigned_nodes = set()
    for vnode in v_order:
        # TOPSIS based ranking for substrate network
        sn_rank = get_ranks_substrate(substrate)
        if node_slice_list is None:
            # Embedding without slicing
            s_order = sn_rank
        else:
            # if slice nodes are less than virtual nodes, return None
            if len(node_slice_list) < virtual.nodes:
                return None
            s_order = [a for a in sn_rank if a in node_slice_list]
        for snode in s_order:
            if substrate.node_weights[snode] >= virtual.node_weights[vnode] and substrate.security_probability[snode] >= virtual.security_probability[vnode] and snode not in assigned_nodes:
                embed_map[vnode] = snode
                substrate.node_weights[snode] -= virtual.node_weights[vnode]
                assigned_nodes.add(snode)
                break

            if snode in assigned_nodes:
                continue

            if substrate.node_weights[snode] < virtual.node_weights[vnode] and substrate.security_probability[snode] < \
                    virtual.security_probability[vnode]:
                reason = "security and CRB requirements."
            elif substrate.security_probability[snode] < virtual.security_probability[vnode]:
                reason = "security requirements."
            elif substrate.node_weights[snode] < virtual.node_weights[vnode]:
                reason = "CRB requirements."

            logging.info(f"Slicing: {vnode} mapping failed on {snode} due to {reason}")
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
    sorder = get_ranks_substrate(substrate)
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
    sorder = get_ranks_substrate(substrate)
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
    edge_slice = dict()

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
        if slice_type == "high" :
            if int(a) in low_slice_node_list or int(b) in low_slice_node_list :
                continue
            edge_slice[nodes] = bandwidth
        elif slice_type == "low" :
            if int(a) in high_slice_node_list or int(b) in high_slice_node_list :
                continue
            edge_slice[nodes] = bandwidth        
    if slice_type == "high" : 
        slice_node_list = high_slice_node_list
    else : 
        slice_node_list = low_slice_node_list

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


def slicing_topsis():
    print(f"\t\t{datetime.now().time()}\tSlicing-Topsis Started")

    # reading substrate and vne from the config files
    substrate, vne_list = pickle.loads(pickle.dumps(config.substrate,-1)), pickle.loads(pickle.dumps(config.vne_list,-1))
    
    # copy of the substrate
    copy_sub = pickle.loads(pickle.dumps(substrate,-1))
    logging.basicConfig(filename="bloc-vne.log", filemode="w", level=logging.INFO)

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
    accepted = 0
    remapped_accepted = 0 # requests accepted after remapping
    revenue = 0
    path_cnt=0
    avg_path_length = 0
    curr_map = dict()  # only contains the requests which are successfully mapped
    pre_resource_edgecost = sum(substrate.edge_weights.values())//2 # total available bandwidth of the physical network
    pre_resource_nodecost = sum(substrate.node_weights.values()) # total crb bandwidth of the physical network
    pre_resource = pre_resource_edgecost + pre_resource_nodecost    
    fault_tolerance_performance = 0

    requests_to_be_remapped = dict() # stores failed embeddings 


    vne_list = sorted(vne_list, key=lambda x: get_revenue(x), reverse=True)  # ascending order
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
        req_map = node_map(pickle.loads(pickle.dumps(substrate,-1)), vne_list[req_no], req_no, slice_node_list)
        if req_map is None:
            logging.info(f"Slicing : Node mapping not possible")
            logging.warning(f"\tNode mapping not possible for req no {req_no}\n")
            # Embedding failed, adding request for non-slicing based embedding
            requests_to_be_remapped[req_no] = vne_list[req_no]
            continue
        logging.info(f"Slicing : Node mapping success for req no {req_no}")       
        req_map = temp_map(vne_list[req_no], req_map)
        if not edge_map(substrate, vne_list[req_no], req_no, req_map, edge_slice):
            logging.info(f"Slicing : Edge mapping not possible for req no {req_no}")
            logging.warning(f"\tEdge mapping not possible for req no {req_no}\n")
            # Embedding failed, adding request for non-slicing based embedding
            requests_to_be_remapped[req_no] = vne_list[req_no]
            continue
        accepted += 1
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
        req_map = node_map(pickle.loads(pickle.dumps(substrate,-1)),virtual_request,req_no)
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
        remapped_accepted += 1
        avg_path_length += findAvgPathLength(virtual_request)
        for edge in req_map.edge_map:
            path_cnt += len(req_map.edge_map[edge])
        req_map.total_cost = req_map.node_cost + req_map.edge_cost
        curr_map[req_no] = req_map
        revenue += sum(vne_list[req_no].node_weights.values()) + sum(vne_list[req_no].edge_weights.values())//2
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
    for edge in substrate.edge_weights:
        post_resource_edgecost += substrate.edge_weights[edge]
        if substrate.edge_weights[edge] != copy_sub.edge_weights[edge]:
            utilized_links += 1
    post_resource_edgecost //= 2
    for node in substrate.node_weights:
        post_resource_nodecost += substrate.node_weights[node]
        if substrate.node_weights[node] != copy_sub.node_weights[node]:
            utilized_nodes += 1

    if accepted != 0:
        avg_path_length /= accepted
    
    post_resource = post_resource_edgecost + post_resource_nodecost

    end_time = datetime.now().time()
    duration = datetime.combine(date.min, end_time) - datetime.combine(date.min, start_time)
    fault_tolerance_performance = accepted/len(req_order) + remapped_accepted/len(req_order)
    logging.info("Start time",start_time)
    logging.info("End time", end_time)
    logging.info("Duration",duration)
    
    logging.info(f"\t\tThe revenue is {revenue} and total cost is {tot_cost}")
    if tot_cost == 0:
        logging.error(f"\t\tCouldn't embed any request")
        output_dict = {
            "revenue": -1,
            "total_cost": -1,
            "accepted": -1,
            "remapped" : -1,
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
            "fault_tolerance_performance" : -1
        }
        print(f"\t\t{datetime.now().time()}\tSlicing-Topsis completed\n")
        return output_dict
    logging.info(f"\t\tThe revenue is {revenue} and total cost is {tot_cost}")
    logging.info(f"\t\tThe fault tolerant performance is {fault_tolerance_performance}")
    logging.info(f"\t\tThe revenue to cost ratio is {(revenue/tot_cost)*100:.4f}%")
    logging.info(f"\t\tTotal number of requests embedded is {accepted} out of {len(vne_list)}")
    logging.info(f"\t\tEmbedding ratio is {(accepted/len(vne_list))*100:.4f}%\n")
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
    # logging.shutdown()
    output_dict = {
        "revenue": revenue,
        "total_cost" : tot_cost,
        "accepted" : accepted,
        "remapped" : remapped_accepted,
        "total_request": len(vne_list),
        "pre_resource": pre_resource,
        "post_resource": post_resource,
        "avg_bw": (ed_cost/pre_resource_edgecost)*100,
        "avg_crb": (no_cost/pre_resource_nodecost)*100,
        "avg_link": (utilized_links/len(substrate.edge_weights))*100,
        "avg_node": (utilized_nodes/len(substrate.node_weights))*100,
        "avg_path": avg_path_length,
        "avg_exec": (duration),
        "total_nodes": total_vnr_nodes,
        "total_links": total_vnr_links,
        "fault_tolerance_performance" : fault_tolerance_performance
    }
    print(f"\t\t{datetime.now().time()}\tSlicing-Topsis completed\n")
    config.re_embed_substrate = pickle.loads(pickle.dumps(substrate,-1)) # copy the current substrate state to be used in re-embedding stage
    return output_dict


if __name__ == '__main__':
    '''
    The output can be different as every time we shuffle the
    request order of VNRs.
    '''

    slicing_topsis()
