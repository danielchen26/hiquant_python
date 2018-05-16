import numpy as np
import numpy.matlib as mtl
import scipy as sp
import random
from scipy.linalg import block_diag
import Solvers
import HIquant_functions as hf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


S = np.array([[1,0],[0,1],[1,1]])
sigma_n = 0.2
M, K = S.shape
N = 6
num_problem = 500



CD_homo_P = dict()
SVD_homo_P = dict()
QP_homo_P = dict()



U_true_set = dict()
for each in range(num_problem):
	Z_true = 1 + np.array([random.expovariate(1) for rand in range(M)])
	U_true = np.random.rand(K,N)
	X_clean = np.diag(Z_true).dot(S).dot(U_true)
	X = X_clean + (X_clean * sigma_n)*np.array(mtl.randn(M, N))

	U_true_set[str(each)] = U_true
	prot_sub =dict()
	prot_sub['S'] = S
	prot_sub['X'] = X
	hf.null_sp_dim(prot_sub)
	# SVD sovler
	SVD_solution = Solvers.SVD(prot_sub)
	SVD_protein, SVD_opt = SVD_solution
	SVD_homo_P[str(each)] = SVD_protein

	# QP solver
	QP_solution = Solvers.QP(prot_sub)
	QP_protein, QP_opt = QP_solution
	QP_homo_P[str(each)] = QP_protein

	# CD sovler
	CD_solution = Solvers.CD(prot_sub, 1000)
	CD_protein, CD_opt = CD_solution
	CD_homo_P[str(each)] = CD_protein


# I should test the correlation between U_true and U_hat
P_CD = np.array(list([np.concatenate(CD_homo_P[str(cc_i)],axis=0) for cc_i in range(num_problem)]))
P_SVD = np.array(list([np.concatenate(SVD_homo_P[str(cc_i)],axis=0) for cc_i in range(num_problem)]))
P_QP = np.array(list([np.concatenate(QP_homo_P[str(cc_i)],axis=0) for cc_i in range(num_problem)]))
P_true = np.array(list([np.concatenate(U_true_set[str(cc_i)],axis=0) for cc_i in range(num_problem)]))

Corr_CD_QP = [np.corrcoef(P_CD[i],P_QP[j])[1][0] for i, j in zip(range(num_problem),range(num_problem))]
Corr_SVD_QP = [np.corrcoef(P_SVD[i],P_QP[j])[1][0] for i, j in zip(range(num_problem),range(num_problem))]
Corr_SVD_CD = [np.corrcoef(P_SVD[i],P_CD[j])[1][0] for i, j in zip(range(num_problem),range(num_problem))]

Corr_CD = [np.corrcoef(P_CD[i],P_true[j])[1][0] for i, j in zip(range(num_problem),range(num_problem))]
Corr_SVD = [np.corrcoef(P_SVD[i],P_true[j])[1][0] for i, j in zip(range(num_problem),range(num_problem))]
Corr_QP = [np.corrcoef(P_QP[i],P_true[j])[1][0] for i, j in zip(range(num_problem),range(num_problem))]
# plt.hist([Corr_CD_QP, Corr_SVD_QP], color=['r','b'], alpha=0.5)
corr_matrix = pd.DataFrame({'Corr: CD-QP':Corr_CD_QP, 'Corr: SVD-CD':Corr_SVD_CD, 'Corr: SVD-QP': Corr_SVD_QP})
inference_corr = pd.DataFrame({'CD_corr':Corr_CD, 'QP_corr':Corr_QP, 'SVD_corr':Corr_SVD})

# a =inference_corr.stack().reset_index(level=1).rename(columns = {'level_1': 'solver', 0 :'correlation'})
corr_matrix.plot(kind='hist', alpha = 0.6, bins = 50,subplots=True)
inference_corr.plot(kind='hist', alpha = 0.8, bins = 50,subplots=True)
plt.show()



# # ----altair module----
# import IPython.display
# def vegify(spec):
#     IPython.display.display({
#         'application/vnd.vegalite.v1+json': spec.to_dict()
#     }, raw=True)
# from altair import Chart
# # ----altair module----
#
#
# # Altair plots for solver accuracy corr[U_true,P_inferred]
# a =inference_corr.stack().reset_index(level=1).rename(columns = {'level_1': 'solver', 0 :'correlation'})
# a
# Corr_hist = Chart(a).mark_bar().encode(
#     x=X('correlation', bin=Bin(maxbins=30)),
#     y='count(*)',
#     color=Color('solver'),
# 	column = 'solver',
# )
# vegify(Corr_hist)
