from pascalanalyzer.pascaldata import PascalData
from pascalanalyzer.pascalmodel import *

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import glob
import warnings
import numpy as np
import pandas as pd

import multiprocessing
from functools import reduce

warnings.simplefilter("ignore")

def process(f):
    df_res= []
    if not f.endswith(".json"): return []
    data= PascalData(f)
    df= data.energy()
    df[["cores","frequency"]]= df[["cores","frequency"]].astype(float)
    df= df[df["input"]=="5"]
    df["energy"]= df["ipmi_energy"]/1e3
    df["frequency"]/=1e6
    df= df.sort_values(["cores","frequency"])
    
    equation = LeastSquaresOptmizer(
"""
pw_eq= lambda x,f,p: (x[0]*f**3+x[1]*f)*p+x[2]
perf_eq= lambda x,f,p: (x[0]*p-x[1]*(p-1))/(f*p)
model= lambda x,p,f: pw_eq(x,f,p)*perf_eq(x[3:],f,p)
""", 6)
        
    models= [
        equation,
        # MLP
        make_pipeline(
        StandardScaler(),
        GridSearchCV(MLPRegressor(max_iter=2000, random_state=0, solver="lbfgs"),
                     {#"hidden_layer_sizes":[100,300,500],
                      "activation":["logistic", "tanh", "relu"],
                       #"solver":["sgd","adam","lbfgs"],
                       "alpha": 10.0**-np.arange(1, 7)})
        ),
        # SVR
        make_pipeline(
        StandardScaler(),
        GridSearchCV(SVR("rbf"),
                     {"C":[1,10,1e3,1e4],
                       "gamma":["auto",0.001,0.01,0.1,1]
                       })
        ),
        make_pipeline(
        SVR("rbf", C=1e4)
        ),
        DecisionTreeRegressor(),
        KNeighborsRegressor()
    ]
    names=["Equation", "MLP", "SVR_gridsearch", "SVR", "Tree", "KNN"]
    for model,name in zip(models,names):
        for ts in [8,16,32,64,128]:
            try:
                fit_model= create_model(df,
                                    inputs=["cores","frequency"],
                                    output="energy",
                                    model=model,
                                    config=data.config,
                                    train_sz=ts,split_type="random")
                err= fit_model.predict(df[["cores","frequency"]].values)-df["energy"]
                mse= sum(err**2)/len(err)
                mae= sum(abs(err)/df["energy"])/len(err)
                df_res.append([f.split("/")[-1], name, ts, mse, mae])
                print(f.split("/")[-1], name, ts, mse, mae)
            except:
                print(f)
    return df_res

if __name__ == "__main__":
    p= multiprocessing.Pool(64)
    r= p.map_async(process,glob.glob("data/*"))
    r.wait()
    df= reduce(lambda x,y:x+y, r.get())
    df= pd.DataFrame(df, columns=["file","name","ts","mse","mae"])
    df.to_csv("data03_random.csv")
