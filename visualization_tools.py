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
# ----altair module----
import IPython.display
def vegify(spec):
    IPython.display.display({
        'application/vnd.vegalite.v1+json': spec.to_dict()
    }, raw=True)
from altair import Chart, load_dataset
# ----altair module----

#
# # Histogram
# def alt_hist(df, var,color_v):
#     hist = Chart(df).mark_bar().encode(
#     x=X(str(var), bin=Bin(maxbins=30)),
#         y='count(*)',
#         color=Color(str(color_v)),
#     	# column = 'solver'
#     )
#     return vegify(hist)
