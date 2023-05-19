# import copy
import os
from time import sleep
# import random
import pandas as pd
# import pprint
# from vikor import main as vikor
from slicing_topsis import slicing_topsis
from slicing_topsis_reembedding import reembedding
from slicing_topsis_dynamic import slicing_topsis_dynamic
from greedy import main as greedy
from topsis_updated import main as topsis
from nrm import main as nrm
from Rematch_AHP import main as rematch_ahp
from rethinking import main as rethinking
from Puvnp import main as Puvnp
from parserr import main as parser
from baseline_attack import attack


from vne_u import create_vne as vne_u
from vne_p import create_vne as vne_p
# from vne_n import create_vne as vne_n
# import graph_extraction_random
import graph_extraction_uniform
import graph_extraction_poisson
# import graph_extraction_normal
import logging
import config
import pickle

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    # formatter = logging.Formatter('%(asctime)s %(levelname)s  : %(message)s')
    formatter = logging.Formatter('[%(levelname)s] : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler) 

output_dict = {
        "algorithm": [''],
        "revenue": [''],
        "total_cost" : [''],
        "revenuetocostratio":[''],
        "accepted" : [''],
        "slicing_accepted" : [''],
        "remapping_accepted" : [''],
        "dynamic_accepted" : [''],
        "total_request": [''],
        "embeddingratio":[''],
        "attacked" : [''],
        "recovered" : [''],
        "recovery_ratio" : [''],
        'fault_tolerance_performance' : [''],
        "pre_resource": [''],
        "post_resource": [''],
        "consumed":[''],
        "avg_bw": [''],
        "avg_crb": [''],
        "avg_link": [''],
        "no_of_links_used": [''],
        "avg_node": [''],
        "no_of_nodes_used": [''],
        "avg_path": [''],
        "avg_exec": [''],
        "total_nodes": [''],
        "total_links": [''],
}

def exec_algo(tot=1,algorithm="greedy") :

    if algorithm == "parser" :
        algo_output = parser()
        printToExcel(
            algorithm='PAGERANK-STABLE',
            revenue=algo_output[0]['revenue'],
            total_cost=algo_output[0]['total_cost'],
            revenuetocostratio=(algo_output[0]['revenue'] / algo_output[0]['total_cost']) * 100,
            accepted=algo_output[0]['accepted'],
            total_request=algo_output[0]['total_request'],
            embeddingratio=(algo_output[0]['accepted']  / algo_output[0]['total_request']) * 100,
            pre_resource=algo_output[0]['pre_resource'],
            post_resource=algo_output[0]['post_resource'],
            consumed=algo_output[0]['pre_resource'] - algo_output[0]['post_resource'],
            avg_bw=algo_output[0]['avg_bw'],
            avg_crb=algo_output[0]['avg_crb'],
            avg_link=algo_output[0]['avg_link'],
            no_of_links_used=algo_output[0]['No_of_Links_used'],
            avg_node=algo_output[0]['avg_node'],
            no_of_nodes_used=algo_output[0]['No_of_Nodes_used'],
            avg_path=algo_output[0]['avg_path'],
            avg_exec=algo_output[0]['avg_exec'].total_seconds() * 1000 / algo_output[0]['total_request'],
            total_nodes=algo_output[0]['total_nodes'],
            total_links=algo_output[0]['total_links']
        )

        printToExcel(
            algorithm='PAGERANK-DIRECT',
            revenue=algo_output[1]['revenue'],
            total_cost=algo_output[1]['total_cost'],
            revenuetocostratio=(algo_output[1]['revenue'] / algo_output[1]['total_cost']) * 100,
            accepted=algo_output[1]['accepted'],
            total_request=algo_output[1]['total_request'],
            embeddingratio=(algo_output[1]['accepted'] / algo_output[1]['total_request']) * 100,
            pre_resource=algo_output[1]['pre_resource'],
            post_resource=algo_output[1]['post_resource'],
            consumed=algo_output[1]['pre_resource'] - algo_output[1]['post_resource'],
            avg_bw=algo_output[1]['avg_bw'],
            avg_crb=algo_output[1]['avg_crb'],
            avg_link=algo_output[1]['avg_link'],
            no_of_links_used=algo_output[1]['No_of_Links_used'],
            avg_node=algo_output[1]['avg_node'],
            no_of_nodes_used=algo_output[1]['No_of_Nodes_used'],
            avg_path=algo_output[1]['avg_path'],
            avg_exec=algo_output[1]['avg_exec'].total_seconds() * 1000 / algo_output[1]['total_request'],
            total_nodes=algo_output[1]['total_nodes'],
            total_links=algo_output[1]['total_links']
        )
        return

    if algorithm == "greedy" :
        algo_output = greedy()
    elif algorithm == "slicing_topsis" :
        algo_output = slicing_topsis()
        printToExcel(
            algorithm=algorithm,
            revenue=algo_output['revenue'],
            total_cost=algo_output['total_cost'],
            revenuetocostratio=(algo_output['revenue']/algo_output['total_cost'])*100,
            accepted=algo_output['accepted'],
            remapped = algo_output['remapped'], # additional metric
            total_request=algo_output['total_request'],
            embeddingratio=((algo_output['accepted'] + algo_output['remapped'])/algo_output['total_request'])*100,
            pre_resource=algo_output['pre_resource'],
            post_resource=algo_output['post_resource'],
            consumed=algo_output['pre_resource']-algo_output['post_resource'],
            avg_bw=algo_output['avg_bw'],
            avg_crb=algo_output['avg_crb'],
            avg_link=algo_output['avg_link'],
            no_of_links_used=algo_output['No_of_Links_used'],
            avg_node=algo_output['avg_node'],
            no_of_nodes_used=algo_output['No_of_Nodes_used'],
            avg_path=algo_output['avg_path'],
            avg_exec=algo_output['avg_exec'].total_seconds()/algo_output['total_request'],
            total_nodes=algo_output['total_nodes'],
            total_links=algo_output['total_links'],
            fault_tolerance_performance=algo_output['fault_tolerance_performance']
        )
        return
    elif algorithm == "slicing_topsis_dynamic" : 
        algo_output = slicing_topsis_dynamic()
        printToExcel(
            algorithm=algorithm,
            revenue=algo_output['revenue'],
            total_cost=algo_output['total_cost'],
            revenuetocostratio=(algo_output['revenue']/algo_output['total_cost'])*100,
            accepted = algo_output['accepted'], # total accepted
            slicing_accepted=algo_output['slicing_accepted'], # VNRs accepted during slicing
            remapping_accepted = algo_output['remapping_accepted'], # VNRs accepted during remapping
            dynamic_accepted = algo_output['dynamic_accepted'], # VNRs dynamically changed and accepted
            total_request=algo_output['total_request'],
            embeddingratio=(algo_output['accepted']/algo_output['total_request'])*100,
            pre_resource=algo_output['pre_resource'],
            post_resource=algo_output['post_resource'],
            consumed=algo_output['pre_resource']-algo_output['post_resource'],
            avg_bw=algo_output['avg_bw'],
            avg_crb=algo_output['avg_crb'],
            avg_link=algo_output['avg_link'],
            no_of_links_used=algo_output['No_of_Links_used'],
            avg_node=algo_output['avg_node'],
            no_of_nodes_used=algo_output['No_of_Nodes_used'],
            avg_path=algo_output['avg_path'],
            avg_exec=algo_output['avg_exec'].total_seconds()/algo_output['total_request'],
            total_nodes=algo_output['total_nodes'],
            total_links=algo_output['total_links'],
        )
        return
    elif algorithm == "slicing_topsis_reembedding" :
        algo_output = reembedding()
        printToExcel(
            algorithm=algorithm,
            revenue=algo_output['revenue'],
            total_cost=algo_output['total_cost'],
            revenuetocostratio=(algo_output['revenue']/algo_output['total_cost'])*100,
            accepted = algo_output['accepted'], 
            attacked = algo_output['attacked'],
            recovered = algo_output['recovered'],
            recovery_ratio = algo_output['recovery_ratio'],
            total_request=algo_output['total_request'],
            embeddingratio=(algo_output['accepted']/algo_output['total_request'])*100,
            fault_tolerance_performance=algo_output['fault_tolerance_performance'],
            pre_resource=algo_output['pre_resource'],
            post_resource=algo_output['post_resource'],
            consumed=algo_output['pre_resource']-algo_output['post_resource'],
            avg_bw=algo_output['avg_bw'],
            avg_crb=algo_output['avg_crb'],
            avg_link=algo_output['avg_link'],
            no_of_links_used=algo_output['No_of_Links_used'],
            avg_node=algo_output['avg_node'],
            no_of_nodes_used=algo_output['No_of_Nodes_used'],
            avg_path=algo_output['avg_path'],
            avg_exec=algo_output['avg_exec'].total_seconds()/algo_output['total_request'],
            total_nodes=algo_output['total_nodes'],
            total_links=algo_output['total_links'],
        )
        return
    elif algorithm[-6:] == "attack" :
    
        algo_output = attack()
        printToExcel(
            algorithm=algorithm,
            revenue=algo_output['revenue'],
            total_cost=algo_output['total_cost'],
            revenuetocostratio=algo_output['revenuetocostratio'],
            accepted = algo_output['accepted'], 
            attacked = algo_output['attacked'],
            recovered = algo_output['recovered'],
            recovery_ratio = algo_output['recovery_ratio'],
            total_request=algo_output['total_request'],
            embeddingratio=(algo_output['accepted']/algo_output['total_request'])*100,
            pre_resource=algo_output['pre_resource'],
            post_resource=algo_output['post_resource'],
            consumed=algo_output['pre_resource']-algo_output['post_resource'],
            avg_bw=algo_output['avg_bw'],
            avg_crb=algo_output['avg_crb'],
            avg_link=algo_output['avg_link'],
            no_of_links_used=algo_output['No_of_Links_used'],
            avg_node=algo_output['avg_node'],
            no_of_nodes_used=algo_output['No_of_Nodes_used'],
            avg_path=algo_output['avg_path'],
            avg_exec=algo_output['avg_exec'].total_seconds()/algo_output['total_request'],
            total_nodes=algo_output['total_nodes'],
            total_links=algo_output['total_links'],
        )
        return
    elif algorithm == "vikor" :
        algo_output = vikor()
    elif algorithm == "topsis" :
        algo_output = topsis()
    elif algorithm == "EAA" :
        algo_output = EAA()
    elif algorithm == "VRMAP" :
        algo_output = vrmap()
    elif algorithm == "rethinking" :
        algo_output = rethinking()
    elif algorithm == "nrm":
        algo_output = nrm()
    elif algorithm == "rematch_ahp" :
        algo_output = rematch_ahp()
    elif algorithm == "Puvnp" :
        algo_output = Puvnp()
    elif algorithm == "topsis" :
        algo_output = topsis()
    # return
    sleep(tot*1)
    printToExcel(
        algorithm=algorithm,
        revenue=algo_output['revenue'],
        total_cost=algo_output['total_cost'],
        revenuetocostratio=(algo_output['revenue']/algo_output['total_cost'])*100,
        accepted=algo_output['accepted'],
        dynamic_accepted=algo_output['dynamic_accepted'],
        total_request=algo_output['total_request'],
        embeddingratio=(algo_output['accepted']/algo_output['total_request'])*100,
        pre_resource=algo_output['pre_resource'],
        post_resource=algo_output['post_resource'],
        consumed=algo_output['pre_resource']-algo_output['post_resource'],
        avg_bw=algo_output['avg_bw'],
        avg_crb=algo_output['avg_crb'],
        avg_link=algo_output['avg_link'],
        no_of_links_used=algo_output['No_of_Links_used'],
        avg_node=algo_output['avg_node'],
        no_of_nodes_used=algo_output['No_of_Nodes_used'],
        avg_path=algo_output['avg_path'],
        avg_exec=algo_output['avg_exec'].total_seconds()/algo_output['total_request'],
        total_nodes=algo_output['total_nodes'],
        total_links=algo_output['total_links'],
    )



def printToExcel(algorithm='', revenue='', total_cost='', revenuetocostratio='', accepted='', slicing_accepted = '', remapping_accepted = '',dynamic_accepted = '',total_request='', 
embeddingratio='', pre_resource='', post_resource='',consumed='',avg_bw='',avg_crb='',avg_link='', no_of_links_used='',
avg_node='', no_of_nodes_used='', avg_path='',avg_exec='', total_nodes='', total_links='',fault_tolerance_performance = '',attacked = '',recovered='',recovery_ratio = ''):
    
    output_dict["algorithm"].append(algorithm)
    output_dict["revenue"].append(revenue)
    output_dict["total_cost"].append(total_cost)
    output_dict["revenuetocostratio"].append(revenuetocostratio)
    output_dict['slicing_accepted'].append(slicing_accepted)
    output_dict["remapping_accepted"].append(remapping_accepted)
    output_dict['dynamic_accepted'].append(dynamic_accepted)
    output_dict["total_request"].append(total_request)
    output_dict["embeddingratio"].append(embeddingratio)
    output_dict["accepted"].append(accepted)
    output_dict['attacked'].append(attacked)
    output_dict['recovered'].append(recovered)
    output_dict['recovery_ratio'].append(recovery_ratio)
    output_dict["pre_resource"].append(pre_resource)
    output_dict["post_resource"].append(post_resource)
    output_dict["consumed"].append(consumed)
    output_dict["avg_bw"].append(avg_bw)
    output_dict["avg_crb"].append(avg_crb)
    output_dict["avg_link"].append(avg_link)
    output_dict["no_of_links_used"].append(no_of_links_used)
    output_dict["avg_node"].append(avg_node)
    output_dict["no_of_nodes_used"].append(no_of_nodes_used)
    output_dict["avg_path"].append(avg_path)
    output_dict["avg_exec"].append(avg_exec)
    output_dict["total_nodes"].append(total_nodes)
    output_dict["total_links"].append(total_links)
    output_dict['fault_tolerance_performance'].append(fault_tolerance_performance)
    addToExcel()


def addToExcel():
    geeky_file = open('output_file.pickle', 'wb')
    pickle.dump(output_dict, geeky_file)
    geeky_file.close()



def generateSubstrate(for_automate, pickle_name):
    substrate, vne = for_automate(3)
    output_file = open(pickle_name, 'wb')
    vne_output = open('1_poisson_vne.pickle','wb')
    pickle.dump(vne,vne_output)
    pickle.dump(substrate, output_file)
    output_file.close()
    vne_output.close()

def extractSubstrate(pickle_file):
    filehandler = open(pickle_file, 'rb')
    substrate = pickle.load(filehandler)
    return substrate



def runExtraction(pickle_name,extraction_method='POISSON'):
    substrate = extractSubstrate(str(pickle_name))
    # vne+ = extractVNE(str(pickle_name))
    # # printToExcel()
    # for _ in range(3):
    #     printToExcel(pre_resource=extraction_method)
    # # printToExcel()
    print(f"\n{extraction_method} Extraction\n")

    if extraction_method == 'POISSON':
        main(substrate, vne_p)
    elif extraction_method == 'RANDOM' :
        main(substrate,vne_r)
    elif extraction_method == 'UNIFORM' :
        main(substrate,vne_u)
    elif extraction_method == 'NORMAL' :
        main(substrate,vne_n)

def main(substrate, vne):
    tot = 0
    number_of_iterations = 2
    config.substrate = pickle.loads(pickle.dumps(substrate,-1))
    for iter in range(number_of_iterations):
        print(f"\n\nIteration {iter+1}\n\n")
        tot = 0
        # sampling 3 vnrs for each iteration 
        tot +=1
        request_counts = [250,500,750,1000]
        for request_count in request_counts:
            vne_list = vne(no_requests=request_count)
            config.vne_list = pickle.loads(pickle.dumps(vne_list,-1))
            exec_algo(tot,"slicing_topsis_dynamic")   # Done
            exec_algo(tot,"slicing_topsis_reembedding") #
            # exec_algo(tot,'greedy') # Done
            # exec_algo(tot,"greedy_attack")
            exec_algo(tot,"topsis")
            exec_algo(tot,"topsis_attack")
            # exec_algo(tot,"nrm")
            # exec_algo(tot,"nrm_attack")
            # exec_algo(tot,"rematch_ahp")
            # exec_algo(tot,"rematch_ahp_attack")
            # exec_algo(tot,"rethinking")
            # exec_algo(tot,"rethinking_attack")
            # exec_algo(tot,"Puvnp") #nOT wORKING
            # exec_algo(tot,"parser")
        printToExcel()


if __name__ == "__main__":

    file_exists = os.path.exists('1_random.pickle') or os.path.exists('1_uniform.pickle') or os.path.exists('1_poisson.pickle') or os.path.exists('1_normal.pickle')
    # file_exists = False
    if not file_exists:
        print("NOT EXIST")
        # generateSubstrate(graph_extraction_random.for_automate, str(1)+'_random.pickle')        #Random Distribution
        # generateSubstrate(graph_extraction_normal.for_automate, str(1)+'_normal.pickle')    #Normal Distribution
        # generateSubstrate(graph_extraction_uniform.for_automate, str(1)+'_uniform.pickle')    #Uniform Distribution
        generateSubstrate(graph_extraction_poisson.for_automate, str(1)+'_poisson.pickle')    #Poission Distribution
    runExtraction("1_poisson.pickle","POISSON")
    # runExtraction('1_uniform.pickle',"UNIFORM")
    excel = pd.DataFrame(output_dict)
    excel.to_excel("Results_Combined.xlsx")
