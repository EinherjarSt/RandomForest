# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import numpy as np
import matplotlib as plot
import pandas as pd


class DecisionTree:
    def __init__(self, training_set, target_var):
        self.training_set = training_set
        self.target_var = target_var
        #print(self.training_set)
        self.__start__()
        
    def __global_entropy__(self, target_name):
        """ Calcula la entropia de la variable objetivo
            Parametro:
                target_name: Nombre de la columna objetivo
        """
        debug = False
        target_column = self.training_set[target_name]
        # calcula la probabilidad elem/total y las deja en un np array
        probability = (target_column.value_counts()/len(target_column)).values
        # aplica logaritmo de 2 a cada elemento
        log_2 =  np.log2(probability)
        entropy = -(probability @ log_2)
        
        if(debug):
            print("\ncount\n", target_column.value_counts())
            print("\ntotal:", len(target_column))
            print("\nprob\n", probability)
            print("\nlog\n", log_2)
            print("\nresult:", entropy)
        return entropy

    def __entropy_2__(self, var_name):
        """ Devuelve un arreglo con las entropias de todas las clases de la variable
            Parametro: Arreglo sobre el cual se sacara la entropia
        """
        debug = False
        
        # Toma la columna objetivo y la columna adicional
        columns = self.training_set[[var_name,self.target_var]]
        
        # Las agrupa por columna adicional(var_name) y la columna objetivo
        var = columns.groupby([var_name,self.target_var])
        
        # Obtiene una tabla que indica la cantidad de elementos 
        # en la columna objetivo por cada clase de la columna adicional
        table = var.size().unstack(fill_value=0)
        
        # Obtiene los indices de las tablas
        index = table.index
        
        # Obtiene la cantidad de registros en la tabla
        vertical_size = table.shape[0]
        entropy_array = []
        if debug:
            print(table, "\n")
            print(vertical_size, "\n")
            print(var_name, "\n")
        i = 0
        while (i < vertical_size):
            #Selecciona la fila i de la tabla
            numerators = table.iloc[i,:]
            
            # Obtiene los valores de la columna que son distinto de 0 como un arreglo numpy 
            numerators = numerators[numerators!= 0].values
            
            # Suma los valores del arreglo para usar como denominador
            denominator = table.iloc[i,:].sum()
            
            # Realiza una division elemento por elemento
            division = numerators / denominator
            
            # Aplica el log2 a cada elemento
            log = np.log2(division)
            
            # Realiza una multiplicacion matricial de vectores.
            # Devido a que son vectores y no matrices el resultado es el mismo
            # que en el producto punto
            entropy = -(division @ log)
      
            if (debug):
                print("\nraw\n",numerators)
                print("\ndenominador\n", denominator)
                print("\ndivision\n",division)
                print("\nlog\n", log)
                print("\nentropy\n",entropy)
                print(17*"-")

                
            # Arma un arreglo con los resultados
            entropy_array += [entropy]
            i += 1;
        # Agrega al arreglo de resultados el indice para indicar a que elemento
        # pertenece la entropia usando pandas
        pandas_entropy = pd.Series(np.array(entropy_array),index= index )
        
        if debug:
            print("\nentropy",var_name, "\n",pandas_entropy)
        return pandas_entropy
            

    
    def __gain__(self, var_name):
        """ Calcula la ganancia que se tendria con una variable
            Parametros:
                var_name: Nombre de la variable
        """
        debug = False
        # Obtiene la entropia de la variable objetivo
        entropy_global = self.__global_entropy__(self.target_var)
        # Obtiene la entropia de la variable var_name
        entropy_array = self.__entropy_2__(var_name)
        # Obtiene los numeradores de la proporcion de la formula de la ganancia
        numerators = self.training_set[var_name].value_counts().sort_index()
        # Obtiene el denominador
        denominators = len(self.training_set)
        proportions = numerators/denominators
        
        # Obtiene ganancia
        ganancia = entropy_global - (proportions @ entropy_array)
        
        if debug:
            print("\nNumerators\n",numerators)
            print("\nDenominators\n",denominators)
            print("\nproportions \n", proportions)
            print("\nresult multiplicacion\n", proportions @ entropy_array)
            print("\nganancia: ", ganancia)
        return pd.Series([ganancia], index=[var_name])

    def __next_level_():
        pass
        
    def __start__(self):
        debug = False
        # Obtengo todos los nombres de las columnas
        columns = self.training_set.columns
        # Obtengo los nombres de las columnas distintos de la variable objetivo
        names = columns[columns != self.target_var]
        gains = pd.Series()
        for column in names:
            gain = self.__gain__(column);
            gains = gains.append(gain)
        print("\nganacias\n", gains)    
            
        if (debug):
            #print("len \n", len(self.training_set))
            #print("primera fila\n",self.training_set.iloc[0])
            #print("pais \n",self.training_set.iloc[0].iat[0])
            #print("religion \n", self.training_set.iloc[0]["religion"])
            training_set_without_religion = self.training_set.loc[:, self.training_set.columns != "religion"]
            #print("sin religion\n", training_set_without_religion)
            #print("pura religion\n", self.training_set["religion"])
            #print("entropia\n", self.__entropy__(self.training_set["religion"]))
        
class RandomForest:
    def __init__(self, training_set, target_var,ntree):
        self.training_set = training_set
        self.target_var = target_var
        self.ntree = ntree
        self.__start__()    
        
    
    def __predict__(self, evaluation_data):
        pass
    
    def _choose_attribute_(self, nvar):
        """
            Escoge aleatoriamente las variables desde la tabla.
            La tabla debe contener la variable objetivo.
            
            Devuelve una tabla con las variables escogidas junto con la variable objetivo
            en la ultima posición.
            
            Parametros:
                nvar: Cantidad de variables a elegir
        """
        number_of_var = self.training_set.shape[1]
        if (isinstance(self.training_set, pd.DataFrame) and
            (0 < nvar < number_of_var)):
            # Obtiene la tabla sin la columna objetivo
            table_without_target = self.training_set[self.training_set.columns[self.training_set.columns != self.target_var]]
            # Elige al azar las variables y une la columna objetivo
            new_table = table_without_target.sample(n=nvar, axis=1).join(self.training_set[self.target_var])
            return new_table
        
        else:
            raise IndexError("El indice no es valido")
            
            
    def __start__(self):
        debug = False
        tree_var = self._choose_attribute_(3)
        arbol_de_decision = DecisionTree(tree_var, self.target_var)
        if (debug):
            print("RandomForest")
    

def main():
    pd.set_option("display.precision", 17)
    debug = False
    headers = ["name","landmass","zone","area","population","language","religion","bars","stripes","colours","red","green","blue","gold","white","black","orange","mainhue","circles","crosses","saltires","quarters","sunstars","crescent","triangle","icon","animate","text","topleft","botright"]
    table = pd.read_csv("resource/flag.data",names=headers)
    #table = pd.read_csv("resource/iris.csv")
    training_data = table[0:50]
    evaluation_data = table[50:]
    target_var = "religion"
    #target_var = "religion"
    if(debug):
        print("table\n",table)
        print("table\n", training_data)
        print("evaluacion\n",evaluation_data)
    randomForest = RandomForest(training_data, target_var, 1)  
    
if __name__ == "__main__":
    main()