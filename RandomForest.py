# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import numpy as np
import matplotlib as plot
import pandas as pd


class DecisionTree:
    def __init__(self, training_set):
        self.training_set = training_set
        #print(self.training_set)
        self.__start__()
        
    def __entropy__(self, target_column):
        debug = False
        print("count\n", target_column.value_counts())
        # calcula la probabilidad elem/total y las deja en un np array
        probability = (target_column.value_counts()/len(target_column)).values
        # aplica logaritmo de 2 a cada elemento
        log_2 =  np.log2(probability)
        entropy = -(probability @ log_2)
        if(debug):
            print("entropia\n", probability)
            print("entropia\n", log_2)
            print("result\n", type(entropy))
        return entropy

    
    def __gain__(self):
        pass
        
    def __start__(self):
        debug = ~False
        if (debug):
            print("len \n", len(self.training_set))
            print("primera fila\n",self.training_set.iloc[0])
            print("pais \n",self.training_set.iloc[0].iat[0])
            print("religion \n", self.training_set.iloc[0]["religion"])
            training_set_without_religion = self.training_set.loc[:, self.training_set.columns != "religion"]
            print("sin religion\n", training_set_without_religion)
            print("pura religion\n", self.training_set["religion"])
            print("entropia\n", self.__entropy__(self.training_set["religion"]))
        
class RandomForest:
    def __init__(self, training_set, out_bag, ntree):
        self.training_set = training_set
        self.out_bag = out_bag
        self.ntree = ntree
        self.__start__()    
        
    
    def __predict__(self, evaluation_data):
        pass
    
    def __start__(self):
        debug = False
        arbol_de_decision = DecisionTree(self.training_set)
        if (debug):
            print(arbol_de_decision)
    
def baggings(data_table):
    debug = False
    training = data_table
    # Toma 50 filas al azar y permite la repeticion
    in_bag = training.sample(n=50, replace=True)
    # Obtiene las filas que no estan en in_bag
    out_bag =  pd.concat([training,in_bag]).drop_duplicates(keep=False)
    if (debug):
        print("table\n", data_table )
        print("training\n", training )
        print("in_bag\n",in_bag)
        print("out_bag\n",out_bag)
    return in_bag, out_bag

def start():
    debug = False
    headers = ["name","landmass","zone","area","population","language","religion","bars","stripes","colours","red","green","blue","gold","white","black","orange","mainhue","circles","crosses","saltires","quarters","sunstars","crescent","triangle","icon","animate","text","topleft","botright"]
    table = pd.read_csv("resource/flag.data",names=headers)
    
    in_bag, out_bag = baggings(table[:50])
    evaluation_data = table[50:]
    if(debug):
        print("table\n",table)
        print("evaluacion\n",table[50:])
    randomForest = RandomForest(in_bag, out_bag, 1)  
    
if __name__ == "__main__":
    start()