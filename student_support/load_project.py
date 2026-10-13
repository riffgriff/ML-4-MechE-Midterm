"""Student support only: input import and fixed splits, without fitted answers.
Usage: from load_project import load_project; X,b,encoded,meta=load_project(1)
X: n-by-m quantitative inputs; b: m-by-1 response; encoded: m-by-(p-1)
unscaled quantitative inputs followed by the prescribed category indicators.
All metadata indices are zero-based NumPy indices. See README.
"""
from pathlib import Path
import json
import numpy as np

def load_project(project):
    root=Path(__file__).resolve().parent/'shared'
    meta=json.loads((root/f'project_{project}_metadata.json').read_text())
    encoded=np.loadtxt(root/f'project_{project}_inputs.csv',delimiter=',',ndmin=2)
    b=np.loadtxt(root/f'project_{project}_response.csv',delimiter=',').reshape(-1,1)
    X=encoded[:,:meta['n']].T
    return X,b,encoded,meta

# For a non-nested validation fit:
# X,b,encoded,meta=load_project(2)
# tr=np.array(meta['folds'][0]['train']); va=np.array(meta['folds'][0]['validation'])
# mu=encoded[tr].mean(axis=0); scale=encoded[tr].std(axis=0,ddof=1)
# scale[scale==0]=1
# A_train=np.column_stack([(encoded[tr]-mu)/scale,np.ones(len(tr))])
# A_validation=np.column_stack([(encoded[va]-mu)/scale,np.ones(len(va))])
# Implement your own OLS/ridge fit on A_train and b[tr], then predict A_validation.
# Group 1: use meta['outer'][j]['inner'] for inner tuning; only then predict
# meta['outer'][j]['test']. meta['folds'] is for the final all-data tuning.
