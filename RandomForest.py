# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import numpy as np
import matplotlib as plot
import pandas as pd

# Si cambia esta variable por true imprimira todos los print del programa
debug = False

class DecisionTree:
    # Variable de prueba
    profundidad = 0
    def __init__(self, training_set, target_var):
        self._training_set = training_set
        self._target_var = target_var
        self._parent = None
        self._branches = {}
        # {key, pandas} Guarda la entropias de las variables
        self._dic_entropy = {}
        
        # Variables para debugear
        DecisionTree.profundidad += 1
        # Profundidad de este nodo
        self._profundidad = DecisionTree.profundidad
                
        self.__start__()
        
    def __global_entropy__(self, target_name):
        """ Calcula la entropia de la variable objetivo
            Parametro:
                target_name: Nombre de la columna objetivo
        """
        #debug = False
        target_column = self._training_set[target_name]
        # calcula la probabilidad elem/total y las deja en un np array
        probability = (target_column.value_counts()/len(target_column)).values
        # aplica logaritmo de 2 a cada elemento
        log_2 =  np.log2(probability)
        entropy = -(probability @ log_2)
        
        if(debug):
            print("\n\n__global_entropy__",80*"-")
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
        #debug = False
        if debug : print("\n\n__entropy_2__ ",80*"-")

        # Toma la columna objetivo y la columna adicional
        columns = self._training_set[[var_name,self._target_var]]
        
        # Las agrupa por columna adicional(var_name) y la columna objetivo
        var = columns.groupby([var_name,self._target_var])
        
        # Obtiene una tabla que indica la cantidad de elementos 
        # en la columna objetivo por cada clase de la columna extra
        table = var.size().unstack(fill_value=0)
        
        # Obtiene los indices de las tablas
        index = table.index
        
        # Obtiene la cantidad de registros en la tabla
        vertical_size = table.shape[0]
        entropy_array = []
        if debug:
            print(table, "\n")
            print("tamanio vertical: ",vertical_size, "\n")
            print("Nombre de la variable a evaluar:",var_name, "\n")
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
                print(80*"-")

                
            # Arma un arreglo con los resultados
            entropy_array += [entropy]
            i += 1;
        # Agrega al arreglo de resultados el indice para indicar a que elemento
        # pertenece la entropia usando pandas
        pandas_entropy = pd.Series(np.array(entropy_array), index= index )
        
        # Guardo la ganancia para usarla en la funcion make_tree, asi no tener que calcularla de nuevo
        self._dic_entropy[var_name] = pandas_entropy
        if debug:
            print("\nentropy",var_name,"resumen", "\n",pandas_entropy)
        return pandas_entropy
            

    
    def __gain__(self, var_name):
        """ Calcula la ganancia que se tendria con una variable
            Parametros:
                var_name: Nombre de la variable
        """
        #debug = False
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
            print("\n\n__gain__",80*"-")
            print("\nNumerators\n",numerators)
            print("\nDenominators\n",denominators)
            print("\nproportions \n", proportions)
            print("\nresult multiplicacion\n", proportions @ entropy_array)
            print("\nganancia: ", ganancia)
        return pd.Series([ganancia], index=[var_name])

    def make_tree(self, gains):
        """ Esta funcion va creando sub_arboles de decision que en algun momento tambien llamaran
            a esta funcion.
            
            Parametros: 
                gains: Serie de las ganancias de cada variable obtenida
        """
        #debug = False
        # Si ya no quedan mas variables entonces retorna
        #if gains.empty:
        #    return
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
                # Lo reduce a solo un valor correspondiente a la clase
                result = self._training_set[[max_gain_name, self._target_var]][self._training_set[max_gain_name] == class_].drop_duplicates()
                self._branches[class_] = result["religion"].values[0]
            # Sino crea un nuevo subarbol
            else:
                # Obtiene un subconjunto de entrenamiento filtrando por la clase
                modified_training_set = self._training_set[self._training_set[max_gain_name] == class_]
                # Quita la variable ya utilizada
                modified_training_set = modified_training_set.drop(max_gain_name, axis = 1)
                # Solo seguir si queda una variable mas aparte de la objetivo
                if (modified_training_set.shape[1] != 1):
                    self._branches[class_] = DecisionTree(modified_training_set, self._target_var)
                    if debug:
                        print("\nmodificado\nclass",class_,"\n", modified_training_set)

                    
        # Elimino la variable que contenia las entropias por que ya las use para determinar que valor
        # de la variable objetivo le correspondia a las clases con entropia 0 de la variable con mayor
        # ganancia (La sentencia if)
        del self._dic_entropy
                
        if debug:
            print("\n\nProfundidad retorno",self._profundidad,80*"=")
            print("\n\nmake_tree",80*"-")
            print("\nmax_var_name:",max_gain_name)
            print("\nclasses\n",classes)
            print("\nbranches\n",self._branches)

        
    def __start__(self):
        """ Comienza la ejecucion de la clase. El constructor se encarga de llamar a este
            metodo
        """
        #debug = ~False
        if debug : 
            print("\n\nProfundidad ",self._profundidad,80*"=") 
            print("\ntraining_set\n", self._training_set)

        
        # Obtengo todos los nombres de las columnas
        columns = self._training_set.columns
        # Obtengo los nombres de las columnas distintos de la variable objetivo
        names = columns[columns != self._target_var]
        # Serie vacia a la que se le iran agregando elementos
        gains = pd.Series()
        for column in names:
            if debug: print("\n\nstart ", column, 80*"-")
            gain = self.__gain__(column)
            gains = gains.append(gain)
 
        if (debug):
            print("\n\nstart to make", 80*"-")
            print("\n\nDict entropia\n", self._dic_entropy,"\n\n")
            print("\nganacias resumen\n", gains)
            
        self.make_tree(gains)
        
    def __predict__(self, evaluation_data):
        """ Sigue la prediccion bajando de arbol a subarboles hasta llegar el resultado
            
            Retorno: El elemento encontrado o el valor None si no puede seguir derivando
        """
        if debug:
            print("Profundidad ", self._profundidad, 80*"-")
            print("\nparent: ",self._parent)
            print("branches\n",self._branches)
        try:
            # Obtiene del dato a analizar la clase correspondiente al nodo de este arbol
            branch_value = evaluation_data[self._parent]
            
            # Obtiene el siguiente elemento del diccionario
            next_node = self._branches[branch_value]
            
            # Si es decicion tree entonces lo recorrera
            if (isinstance(next_node, DecisionTree)):
                # Llama a la  funcion de evaluacion del nodo hijo
                result = next_node.__predict__(evaluation_data)
                # Devuelve solo si el resultado no es None
                if (result != None):
                    return result
                else:
                    return None
            # Si es un entero entonces devuelve
            else:
                if debug: print(self._target_var, "es: ", next_node)
                return next_node
        except KeyError:
            if debug: print("No existe ", evaluation_data[self._parent])
            return None
                    
        if debug:
            print("\nProfundidad ", self._profundidad," retorno", 80*"-")
            print("\nvalor de la branch: ", branch_value)
            print("\nnext_nodo\n",next_node)

        

        
class RandomForest:
    def __init__(self, training_set, target_var,ntree, nvar):
        self._training_set = training_set
        self._target_var = target_var
        self._ntree = ntree
        self._nvar = nvar
        self._list_of_trees = []
        self.__start__()
        
    
    def __predict__(self, evaluation_data):
        """ Evalua el dato pasado y retorna un entero con el resultado
            Parametro:
                evaluation_data: Instancia(elemento de la fila) de dato en formato pandas
                
            retorna: Un resultado del tipo que sea la variable objetivo
        """
        if debug: print(evaluation_data)
        counts = pd.Series();
        for tree in self._list_of_trees:
            # Consulta el valor de la prediccion
            result = tree.__predict__(evaluation_data)
            # Si el resultado no es None lo agrega a la serie
            if result != None:
                counts = counts.append(pd.Series(result))
                
        if (counts.value_counts().empty):
            return None
        else:
            # Cuenta la cantidad de elementos de cada tipo y retorna el mas repetido
            return counts.value_counts().idxmax()
    
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
        """ Comienza la ejecucion de la clase. El constructor se encarga de llamar a este
            metodo
        """
        debug = False
        i = 0
        while (i < self._ntree):
            # Escoge aleatoriamente los atributos para cada arbol que cree
            tree_var = self._choose_attribute_(self._nvar)
            # Agrega el arbol creado a una lista
            self._list_of_trees += [DecisionTree(tree_var, self._target_var)]
            i += 1
            
        if (debug):
            print("RandomForest")
            print(self._list_of_trees)
    

def random(table, nrandom):
    """ Escoge al azar variables para armar el arbol
    """
    if 0 > nrandom or nrandom > len(table):
        raise IndexError("El indice no es valido")
    
    # Escoge al azar nrandom variables
    training_data = table.sample(n=50)
    # Obtiene las variables que no escogio
    evaluation_data = pd.concat([table,training_data]).drop_duplicates(keep=False)
    return training_data, evaluation_data
    
    
def main():
    # Configuraciones de imprecion pandas
    pd.set_option("display.precision", 17)
    pd.options.display.max_columns = 2000
    debug = False
    headers = ["name","landmass","zone","area","population","language","religion","bars","stripes","colours","red","green","blue","gold","white","black","orange","mainhue","circles","crosses","saltires","quarters","sunstars","crescent","triangle","icon","animate","text","topleft","botright"]
    table = pd.read_csv("resource/flag.data",names=headers)
    
    #training_data = table[0:50]
    #evaluation_data = table[50:]
    training_data, evaluation_data = random(table, 100)
    target_var = "religion"
    
    if(debug):
        print("table\n",table)
        print("table\n", training_data)
        print("evaluacion\n",evaluation_data)
    
    randomForest = RandomForest(training_data, target_var, 400, 4)
    
    if(debug): print("\n\n evaluacion",80*"-")
    sucess_rate = 0
    for i in range(len(evaluation_data)):
        print(80*"-")
        test_data = evaluation_data.iloc[i,:]
        print(repr(evaluation_data.iloc[[i]]))
        expected_result = test_data[target_var]
        test_data = test_data.drop(target_var)
        result = randomForest.__predict__(test_data)
        print("\n\nresultado prediccion: ", result)
        print("resultado esperado: ", expected_result)
        if (result == expected_result): sucess_rate += 1
    print("\n\n sucess_rate: {:.2%}%".format(sucess_rate / len(evaluation_data)))

if __name__ == "__main__":
    main()