import logging
import config 
import sys
import pickle
import random
from datetime import datetime, date

def attack() :
    print(f"\t\t{datetime.now().time()}\tBasline Server Attack Started")
    
    substrate = pickle.loads(pickle.dumps(config.re_embed_substrate,-1))
    copy_sub = pickle.loads(pickle.dumps(config.substrate,-1))
    vne_list = pickle.loads(pickle.dumps(config.re_embed_vne_list,-1))
    request_mappings = pickle.loads(pickle.dumps(config.curr_map,-1))
    active_servers = pickle.loads(pickle.dumps(config.active_servers,-1))
    previous_accepted = substrate.successful_requests

    logging.basicConfig(filename="baseline-reembedding.log", filemode="w", level=logging.INFO)
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
    final_reembed_mapping = request_mappings
    # Iterate through all VNRs

    # Fetching revenue from previous module
    for req_no in request_mappings :
        virtual = vne_list[req_no]
        revenue += sum(vne_list[req_no].node_weights.values()) + (sum(vne_list[req_no].edge_weights.values())//2)
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
                
        # If there are no VMs on attacked servers
        if len(vms_to_be_embedded) == 0 :
            logging.info(f"No re-embedding required for {req_no}")
            continue
        vnrs_attacked += 1
        # Release node resources
        for vm,server in enumerate(curr_map.node_map):
            substrate.node_weights[server] += virtual.node_weights[vm]
        # Release link resources
        if isinstance(curr_map.edge_map,list) :
            # Release resources for rethinking
            for i, path in enumerate(curr_map.edge_map):
                for j in range(1, len(path)):
                    substrate.edge_weights[
                        (str(path[j - 1]), str(path[j]))
                    ] += curr_map.edge_weight[i]
                    substrate.edge_weights[
                        (str(path[j]), str(path[j-1 ]))
                    ] += curr_map.edge_weight[i]
        else : 
            for edge, path in curr_map.edge_map.items():
                edge = edge[1]
                for i in range(1, len(path)):
                    substrate.edge_weights[(path[i - 1], path[i])] += virtual.edge_weights[edge]
                    substrate.edge_weights[(path[i], path[i-1])] += virtual.edge_weights[edge]
        # Release link resources
        revenue -= sum(vne_list[req_no].node_weights.values()) + sum(vne_list[req_no].edge_weights.values())//2
        total_vnr_nodes += virtual.nodes
        total_vnr_links += len(virtual.edges) 
        del final_reembed_mapping[req_no]

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
    
    post_resource = post_resource_edgecost + post_resource_nodecost
        
    logging.info(f"Total attacked VNRs : {vnrs_attacked}")
    logging.info(f"Total successfully re-embedded VNRs : {vnrs_reembed}")

    end_time = datetime.now().time()
    duration = datetime.combine(date.min, end_time) - datetime.combine(date.min, start_time)
    if vnrs_attacked == 0 :
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
            "revenuetocostratio" : -1,
            "attacked" : -1,
            "recovered" : -1,
            "recovery_ratio" : -1,
        }
        print(f"\t\t{datetime.now().time()}\tBaseline Attack completed\n")
        return output_dict
    if tot_cost == 0 :
        revenuetocostratio = -1
    else : 
        revenuetocostratio = (revenue / tot_cost) * 100
    output_dict = {
            "revenue": revenue,
            "total_cost": tot_cost,
            "accepted": total_accepted,
            "total_request": len(vne_list),
            "revenuetocostratio" : revenuetocostratio,
            "attacked" : vnrs_attacked,
            "recovered" : vnrs_reembed,
            "recovery_ratio" : (vnrs_reembed/vnrs_attacked)*100,
            "pre_resource": pre_resource,
            "post_resource": post_resource,
            "avg_bw": (average_edge_utilization) * 100,
            "avg_crb": (average_node_utilization) * 100,
            "avg_link": ((utilized_links / len(substrate.edge_weights)) / 2) * 100,
            "No_of_Links_used": (utilized_links // 2),
            "avg_node": (utilized_nodes / len(substrate.node_weights)) * 100,
            "No_of_Nodes_used": (utilized_nodes),
            "avg_path": avg_path_length,
            "avg_exec": duration,
            "total_nodes": total_vnr_nodes,
            "total_links": total_vnr_links
    }
    print(f"\t\t{datetime.now().time()}\tBaseline Server Attack completed\n")
    return output_dict

if __name__ == '__main__':
    attack()
