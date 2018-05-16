Research


- Stack inferred protein matrix to a single vector along row dimension.
  ``` python
  P_homo_SVD_i = np.concatenate(SVD_protein, axis=0) # stack rows of inferred SVD_protein to a vector
  P_homo_SVD[str(cc_i)] =  P_homo_SVD_i

  P_homo_QP_i = np.concatenate(QP_protein,axis=0) # stack rows of inferred QP_protein to a vector
  P_homo_QP[str(cc_i)] = P_homo_QP_i

  P_homo_CD_i = np.concatenate(CD_protein,axis=0) # stack rows of inferred CD_protein to a vector
  P_homo_CD[str(cc_i)] = P_homo_CD_i
  ```
- Another way of stacking the matrix row-wise after the looping is
  ``` python
  mask = np.concatenate(non_empty_ind >0)[0:21]
  # Get the dictionary values and Select the homologous group
  # ♌ Need to use dictionary values directly because not all
  # ♌ homo_group has same number of proteins, therefore result
  # ♌ in diffferent vector lengths
  P_CD1 = np.array(list([np.concatenate(CD_homo_P[str(cc_i)],axis=0) for cc_i in range(21)]))[mask]
  P_SVD1 = np.array(list([np.concatenate(SVD_homo_P[str(cc_i)],axis=0) for cc_i in range(21)]))[mask]
  P_QP1 = np.array(list([np.concatenate(QP_homo_P[str(cc_i)],axis=0) for cc_i in range(21)]))[mask]
  ```
- numpy.matlib return data type 'matrix' instead of array
- ​
