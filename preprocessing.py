import pandas as pd
import numpy as np

class preprocessing:
    def __init__(self, file_name, show_preprocessed=0):
        try:
            data_brut = pd.read_csv(file_name)
            data = data_brut.to_numpy()
            X = data[:, 2:]
            Y = data[:, 1]
        except Exception as e:
            print("Error : ", e)
            exit()

        self.X = self.get_normalize_data(X)

        self.res_dic = {}
        possible_result = set(Y)
        for i, res in enumerate(possible_result):
            self.res_dic[i] = res
        
        self.Y = [1 if y == "M" else 0 for y in Y]
        self.nb_output = 1 if len(possible_result) == 2 else len(possible_result)
        self.nb_features = np.shape(X)[1]

        if show_preprocessed == 1:
            print("\033[1m" + "\nPreprocessing : " + "\033[0m")
            print("Resultats possible : ", self.res_dic)
            print("Echantillon :", len(Y))
            print("Nombre de features :", self.nb_features)
            print("Nombre d'output possible :", self.nb_output)

    def get_normalize_data(self, data):

        def normalize(data, max, min):
            return (data - min) / (max - min) * 100

        for column in range(data.shape[1]):
            min = np.min(data[:, column])
            max = np.max(data[:, column])
            for row in range(data.shape[0]):
                data[row][column] = normalize(data[row][column], max, min)
        return data

