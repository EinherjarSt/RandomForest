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
        self._training_set = training_set
        self._target_var = target_var
        self._parent = None
        self._branches = {}
        # {key, pandas} Guarda la entropias de las variables
        self._dic_entropy = {}

        self.__start__()
        
    def __global_entropy__(self, target_name):
        """ Calcula la entropia de la variable objetivo
            Parametro:
                target_name: Nombre de la columna objetivo
        """
        debug = False
        target_column = self._training_set[target_name]
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
        columns = self._training_set[[var_name,self._target_var]]
        
        # Las agrupa por columna adicional(var_name) y la columna objetivo
        var = columns.groupby([var_name,self._target_var])
        
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
        pandas_entropy = pd.Series(np.array(entropy_array), index= index )
        
        self._dic_entropy[var_name] = pandas_entropy
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
        entropy_global = self.__global_entropy__(self._target_var)
        # Obtiene la entropia de la variable var_name
        entropy_array = self.__entropy_2__(var_name)
        # Obtiene los numeradores de la proporcion de la formula de la ganancia
        numerators = self._training_set[var_name].value_counts().sort_index()
        # Obtiene el denominador
        denominators = len(self._training_set)
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

    def make_tree(self, gains):
        """ Esta funcion va creando sub_arboles de decision.
            
            Parametros: 
                gains: Serie de las ganancias de cada variable obtenida
        """
        debug = ~False
        if gains.empty:
            return
        # Obtiene la variable que tiene la mayor ganancia
        max_gain_name = gains.idxmax()
        # Establece la variable de mayor ganancia como la raiz de este arbol
        self._parent = max_gain_name
        # Obtiene todas las clases de la variable con mayor ganancia
        classes = self._training_set[max_gain_name].drop_duplicates()
        # Recorre y agrega las ramas y nuevos nodos
        for class_ in classes:
            # Si la entropia es 0 agrega la religion directamente
            if(self._dic_entropy[max_gain_name][class_] == 0):
                r = self._training_set[[max_gain_name, self._target_var]][self._training_set[max_gain_name] == class_].drop_duplicates()
                self._branches[class_] = r["religion"].values[0]
            # Sino crea un nuevo subarbol
            else:
                # Obtiene un subconjunto de entrenamiento filtrando por la clase
                modified_training_set = self._training_set[self._training_set[max_gain_name] == class_]
                # Quita la variable ya utilizada
                modified_training_set = modified_training_set.drop(max_gain_name, axis = 1)
                self._branches[class_] = DecisionTree(modified_training_set, self._target_var)
                if debug:
                    print("\nmodificado\nclass",class_,"\n", modified_training_set)

                
        if debug:
            print("\nmax_var_name:",max_gain_name)
            print("\nclasses\n",classes)
            print("\nbranches\n",self._branches)
            print("\ndic_entropy\n", self._dic_entropy)

        
    def __start__(self):
        debug = ~False
        # Obtengo todos los nombres de las columnas
        columns = self._training_set.columns
        # Obtengo los nombres de las columnas distintos de la variable objetivo
        names = columns[columns != self._target_var]
        gains = pd.Series()
        for column in names:
            gain = self.__gain__(column);
            gains = gains.append(gain)
 
        if (debug):
            print("\nganacias\n", gains)
            
        self.make_tree(gains)

        
class RandomForest:
    def __init__(self, training_set, target_var,ntree):
        self._training_set = training_set
        self._target_var = target_var
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
        number_of_var = self._training_set.shape[1]
        if (isinstance(self._training_set, pd.DataFrame) and
            (0 < nvar < number_of_var)):
            # Obtiene la tabla sin la columna objetivo
            table_without_target = self._training_set.drop(self._target_var, axis=1)
            # Elige al azar las variables y une la columna objetivo
            new_table = table_without_target.sample(n=nvar, axis=1).join(self._training_set[self._target_var])
            return new_table
        
        else:
            raise IndexError("El indice no es valido")
            
            
    def __start__(self):
        debug = False
        tree_var = self._choose_attribute_(3)
        arbol_de_decision = DecisionTree(tree_var, self._target_var)
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