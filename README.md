# PriDe
## Execution Environment:

Operation System: Microsoft Windows 10, 64bit.<br />
Physical Memory (RAM) 16.0 GB.<br />


### Prerequisites

Python 3.9.<br />
PyCharm Community Edition 2021.2. <br />
Alib utility for VNE simulation.<br />
An introduction about VNE problem can be found in the below link:<br />
https://www.youtube.com/watch?v=JKB3aVyCMuo&t=506s<br />

### Installation

###  Download the ALIB_Utility tool, unzip and copy it to the execution drive.<br /> 

- Configure the alibi by following the steps mentioned in the GitHub repository [1]. <br />

- Generate the input. pickle file and save it in the P3_ALIB_MASTER\input path. <br />

- Make sure "P3_ALIB_MASTER\input" path contain senario_RedBestel.pickle. If not, generate the substrate network scenario for "senario_RedBestel.pickle" in folder P3_ALIB_MASTER\input, and this pickle file contains substrate network information.<br />

###   Download  PriDe and keep it in the drive where P3_ALIB_MASTER is present. The  PriDe file contains all executable files related to the proposed and baseline approaches. <br />

- slicing_topsis_dynamic.py -> The Main file related to the  Proposed PriDe approach.<br />
- Greedy.py -> The Main file related to the  Greedy baseline approach [7].<br />
- Nrm.py -> The Main file related to the  NRM baseline approach [2]. <br /> 
- Rethinking.py -> The Main file related to the  DPGA baseline approach [3]. <br />
- Rematch.py -> The Main file related to the  ReMatch  baseline approach [6]. <br />




## Usage

###  In vne_u.py, we can set the various parameters related to Virtual network requests(VNRs).<br />

- We can set the minimum and maximum number of VNR VMs in the create_vne function.<br />

- We can set the virtual network request demands like BandWidth(min, max), CRB(min, max), LocationX(min, max), LocationY(min, max), Delay(min, max), Node Security(min,max), Link Security(min,max), RSA_key(min,max), and Priority_bit(0, 1) in vne. append function. <br />
- Example: (1, 5, 1, 10, 0, 100, 0, 100, 1, 4, 5, 12, 5, 12, 10, 20, 0, 1)<br />

- Run vne_u.py after doing any modifications. <br />

###  In grpah_extraction_poission.py:<br />

- In the get_graphs function mention the pickle file related to substrate network generation, the same is available in the folder P3_ALIB_MASTER. EX: os.path.join( os.path.dirname(current), "P3_ALIB_MASTER", "input", "senario_RedBestel.pickle",)<br />

- In graph.parameters function set substrate network resources like BandWidth(min,max), CRB(min,max), LocationX(min,max), LocationY(min,max), Delay(min,max), Node Security(min,max), Link Security(min,max) .<br />
- Example: (500, 1000, 200, 1000, 0, 100, 0, 100, 1, 1, 10, 20, 10, 20, 15)<br />

- Run grpah_extraction_uniform.py after doing any modification. <br />

### grpah_p.py

- This file generates the standard 1_uniform.pickle file, which contains all the information about substrate network topologies, such as the number of servers, links, and connectivity. It also includes values for each substrate network resource.

### In the automate.py file, set the VNR size such as [250, 500, 750, 1000] and also mention the number of iterations needed to execute for each VNR size in the iteration variable.<br />

- Finally, run the automate.py file. After successfully running, a 1_poission.pickle and 1_poisson_vne.pickle file is created related to SN and VNRs, respectively. (If it already does not exist in the specified path). It has all input parameters related to the substrate network parameters, such as CRB, Bandwidth, Delay, and Location.

- Final embedding results are captured in Results.xlsx, which includes values for various metrics for all test scenarios for every iteration.

### References
[1] E. D. Matthias Rost, Alexander Elvers, “Alib,” https://github.com/vnep-approx/alib, 2020. <br />
[2] P. Zhang, H. Yao, Y. Liu, Virtual network embedding based on computing, network, and storage resource constraints, IEEE Internet of Things Journal 5 (5) (2017) 3298–3304. doi: https://doi.org/10.1109/JIOT.2017.2726120. <br />
[3] Nguyen, Khoa TD, Qiao Lu, and Changcheng Huang. "Rethinking virtual link mapping in network virtualization." In 2020 IEEE 92nd Vehicular Technology Conference (VTC2020-Fall), pp. 1-5. IEEE, 2020, https://ieeexplore.ieee.org/document/9348799. <br />
[6] A. Satpathy, M. N. Sahoo, L. Behera, C. Swain, ReMatch: An Efficient Virtual Data Center Re-Matching Strategy Based on Matching Theory,IEEE Transactions on Services Computing (2022). doi: https://doi.org/10.1109/TSC.2022.3183259. <br />
[7] TG, Keerthan Kumar, Sourav Kanti Addya, Anurag Satpathy, and Shashidhar G. Koolagudi. "NORD: NOde Ranking-based efficient virtual network embedding over single Domain substrate networks." Computer Networks 225 (2023): 109661. doi: https://doi.org/10.1016/j.comnet.2023.109661. <br />

## Contributors
- Mr. Keerthan Kumar T G<br />
https://scholar.google.com/citations?user=fW7bzK8AAAAJ&hl=en <br />
- Mr. anurag satpathy<br />
https://anuragsatpathy.github.io/<br />
- Dr. Sourav kanti addya<br />
https://souravkaddya.in/<br />
- Prof. Shashidhar G Koolagudi <br />
https://scholar.google.co.in/citations?user=WAyKHHwAAAAJ&hl=en <br />
- Prof. Sajal K. Das <br />
https://scholar.google.com/citations?user=mbaG-mQAAAAJ&hl=en <br />

## Contact

If you have any questions, simply write a mail to  Keerthanswamy(AT)gmail(DOT)com.
