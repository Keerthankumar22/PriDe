import os
import pickle
import sys
import graph_p
from vne_p import create_vne


class Extract:
    def get_graphs(self, req_no = 5):     # USE THIS DEFINITION FOR AUTOMATION & comment line no 10
        current = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(os.path.join(os.path.dirname(current), "P3_ALIB_MASTER"))
        current = os.path.join(
            # os.path.dirname(current),
            "C:\\",
            "P3_ALIB_MASTER",
            "input",
            "senario_RedBestel.pickle", #"KK_Aarnet.pickle"
        )
        with open(current, "rb") as f:
            data = pickle.load(f)
        # Parameters : Bandwidth, CPU, PositionX, PositionY, Delay, Node Security, Link Security, RSA, Priority
        para = graph_p.Parameters(500, 1000, 200, 1000, 0, 100, 0, 100, 1, 1, 10, 20, 10, 20, 15, 25, 0, 1)
        # para = graph_p.Parameters(50, 100, 20, 100, 0, 100, 0, 100, 1, 1, 10, 19, 10, 19, 15, 25, 0, 1) # Test Sample
        mean = 0.4
        
        substrate = graph_p.Graph(
            len(data.scenario_list[0].substrate.nodes),
            data.scenario_list[0].substrate.edges,
            para,
            # mean,
        )

        vne_list = create_vne(no_requests = req_no)   # USE THIS STATEMENT FOR AUTOMATION & comment line no 28
        return substrate, vne_list

def for_automate(req_no = 5):
    x = Extract()
    substrate, vne_list = x.get_graphs(req_no)
    return substrate, vne_list


if __name__ == "__main__":
    for_automate(req_no=5)