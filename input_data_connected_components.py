import numpy as np
import pandas as pd
import os
import sys
import scipy.sparse as sps
import time
# Data input
test_path = '/Users/chentianchi/Desktop/test/current/data/hiquant_getdata_test/HIquant_test_alex/IFN/all_proteoform/paralogs_extract.txt'
inputdat = pd.read_csv(test_path, sep='\t', header=None)
# Drop rows that contains Nan or 0 values.
inputdat = inputdat.dropna()# matlab: dat.textdata



Textdata = inputdat.iloc[:, [0, 1]]
print('The text data is:\n', Textdata)

Data = inputdat.iloc[:, 2::]
print('The data values are:\n', Data)

# generte paralogs_all_protein.txt file. all unique protein name
Proteins = inputdat.iloc[:,0].str.split(';', expand=True).iloc[:, 0].unique()
print('The protein names are:\n', pd.DataFrame(Proteins))





# Calculate the NO. of proteins
Num_proteins = len(np.unique(Proteins)) # find the unique number of proteins
Num_proteins
Num_peptides = len(inputdat.index) # find the number of peptides
Num_peptides


# test connected components & Making Stochiometry Matrix
print('Constructing Stochiometry Matrix')


st = time.time()
A = sps.lil_matrix((Num_proteins, Num_proteins))
S = np.zeros((Num_peptides, Num_proteins))
Data_isunique = np.zeros((Num_peptides, 1))
df = pd.DataFrame(Proteins)
for pep_i in range(Num_peptides):
    proteins = Textdata.iloc[pep_i, 0].split(';')
    num_prots = len(Textdata.iloc[pep_i, 0].split(';'))
    # print('protein name:',proteins)
    # print('index :', np.where(Proteins == proteins))
    idxlist = []#np.zeros(num_prots)
    for prot_i in range(num_prots):
        p = proteins[prot_i]
        indices, = np.where(np.in1d(Proteins,p))
#         indices = df.index[df.iloc[:,0].isin(proteins)==True].values
        # print('protein indices:', indices)
        S[pep_i, indices] = S[pep_i, indices] + 1
        idxlist.append(indices)
    idxlist = np.unique(idxlist)
    print('final protein list:', idxlist)
    Data_isunique[pep_i] = len(idxlist)
    newid = np.array([[i, j] for i in idxlist for j in idxlist])
    print ('new id:',newid)
    A[newid[:, 0], newid[:, 1]] = 1

ed = time.time()
dura = (ed-st)
print('time lapse:', dura)

# Get connected_components
graph = A
n_components, component_list = sps.csgraph.connected_components(graph)
n_components
component_list

np.where(np.in1d(component_list,15783))
# test A construction
A = sps.lil_matrix((Num_proteins, Num_proteins))
S = np.zeros((Num_peptides, Num_proteins))
for pep_i in range(Num_peptides):
    proteins = Textdata.iloc[pep_i, 0].split(';')
    num_prots = len(Textdata.iloc[pep_i, 0].split(';'))
    # print('protein name:',proteins)
    # print ('number of proteins :', num_prots)
    pro_index = np.where(np.in1d(Proteins, proteins))[0]
    # print('protein index ', pro_index)
    idxlist = np.unique(pro_index)
    # print('the unique index:', idxlist)
    newid = np.array([[i, j] for i in idxlist for j in idxlist])
    print (newid)
    A[newid[:, 0], newid[:, 1]] = 1
    # A[idxlist, idxlist] = 1























# idx =[1,3,4]
# newid = np.array([[i, j] for i in idx for j in idx])
# newid
# newid[:,0]
# newid[:,1]
# matrix[newid[:,0],newid[:,1]]=1
#
# b = []
#
# b.append([1,3])
# b
#
#
# matrix = np.zeros((3,5), dtype = int)
# indices = np.array([[1,3], [2,4], [0,4]])
