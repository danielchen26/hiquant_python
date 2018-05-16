import numpy as np
import pandas as pd
import os
import sys
import scipy.sparse as sps
from scipy.sparse.csgraph import connected_components
import numba
import time
import HIquant_functions as hf
import Solvers
os.getcwd()

print( 'Importing data from the directory below ...')
# Define function to get the input and parse it.
st=time.time()
@numba.jit
def GetData ( path ):
    if len(sys.argv) < 1:
        print (' You have to input the absolute Path of your file! ')
    else:
        Path=path
        print ('The file path you inputed is : \n', Path)
        dat = pd.read_csv(Path,sep='\t',header=None)
        dat = dat.dropna()
        Textdata = dat.iloc[:,[0,1]]
        Data = dat.iloc[:,2::]
        Proteins = dat.iloc[:,0].str.split(';', expand=True).iloc[:, 0].unique()
        print('Proteins :{}\n'.format(Proteins))
        # Check the number of Columns, HIquant need at least 2 columns
        if len(Data.columns) < 2:
            print('HIquant needs at least 2 conditions')
            return
        else:
            print('Number of conditions checked !')

        # find the unique number of proteins
        Num_proteins = len(np.unique(Proteins))
        # find the number of peptides
        Num_peptides = len(dat.index)

        # Making Stochiometry Matrix （wrong）
        print('Constructing Stochiometry Matrix')
        A = sps.lil_matrix((Num_proteins,Num_proteins))
        S = np.zeros((Num_peptides, Num_proteins))
        Data_isunique = np.zeros((Num_peptides,1))
        for pep_i in range(Num_peptides):
            proteins = Textdata.iloc[pep_i,0].split(';')
            num_prots = len(Textdata.iloc[pep_i,0].split(';'))
            idxlist = []
            print('The peptide {} associates {} proteins : {}'.format(pep_i,num_prots,proteins))
            for prot_i in range(num_prots):
                p = proteins[prot_i]
                indices, =np.where(np.in1d(Proteins,p))
                S[pep_i,indices] = S[pep_i,indices] + 1
#                 print('idxlist is :', idxlist)
#                 print('indices is :', indices)
                idxlist.append(indices)#[prot_i]=indices
                print('The protein {} has associated index {}'.format(p,indices))
                print('The S matrix row index:{} column index:{}'.format(pep_i, indices))
                print('The indices {} in the orignal Proteins list is protein:{}'.format(indices,Proteins[indices]))
            idxlist = np.unique (idxlist)
            print('The peptide {} final protein column index list:{}\n'.format(pep_i, idxlist))
            Data_isunique[pep_i] = len(idxlist)
            newid = []
            for i in idxlist:
                for j in idxlist:
                    newid.append([i,j])

            newid = np.array(newid)
#             print ('new id:',newid)
            A[newid[:, 0], newid[:, 1]] = 1


        print ('Stochi matrix finished!')
        n_components, component_list = connected_components(A,directed=True)

        # Returning the following values
        Mat_A = A
        Mat_S = S
        Mat_CC = component_list
        Mat_nC = n_components
        Dat_data = Data
        Dat_textdata = Textdata
        Dat_proteins = Proteins
        return Mat_A , Mat_S, Mat_CC , Mat_nC, Dat_data, Dat_textdata, Dat_proteins, Data_isunique, Num_peptides, Num_proteins

# test_path='/Users/chentianchi/Desktop/test/current/data/hiquant_getdata_test/HIquant_test_alex/alex_kea_hiquant/paralogs_extract.txt'
test_path='/Users/chentianchi/Desktop/test/current/data/hiquant_getdata_test/HIquant_test_alex/IFN/all_proteoform/paralogs_extract.txt'
a=GetData(test_path)
ed=time.time()
print('Time:',ed-st)
# Above is the verification of the get_data function, the order in the connected_components is diffferent from matlab
######################################################---------------------------------##########################################
######################################################---------------------------------##########################################
