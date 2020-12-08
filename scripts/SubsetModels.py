import pandas as pd 
import numpy as np 
from DataSubsetter import DataSubsetter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class SubsetModelTrainer(DataSubsetter):
    
    def __init__(self, df, y, columns, model, comb_size = None, 
                 data_col = None, combs=True, library='statsmodels', 
                 kwargs={}, train_test_split = False, ttprop = 0.2, fit_type = 'fit'):
        DataSubsetter.__init__(self, df, columns, comb_size)
        self.model = model
        self.data_col = data_col
        self.combs = combs
        self.kwargs = kwargs
        self.y = y
        self.train_test_split = train_test_split
        self.ttprop = ttprop
        self.library = library
        self.fit_type = fit_type
        if data_col:
            self.data_col = self.data_col
        else:
            self.data_col = list(df) 
    

    def fitStatsModel(self,x, y):
        #print(y)#.astype(float).value_counts())
        setup = self.model(endog=y.astype(float), exog=x.astype(float))
        if self.fit_type == 'fit':
            trained = setup.fit(**self.kwargs)
        elif self.fit_type == 'fit_regularized':
            trained = setup.fit_regularized(**self.kwargs)

        return trained

    def fitSciKit(self, x, y):
        # This might not work
        model = self.model(**self.kwargs)

        model.fit(x, y)
        return model 

    def modTest(self, x, y):
        if self.train_test_split:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.ttprop)
        else: 
            X_train = X_test = x
            y_train = y_test = y
        if self.library == 'statsmodels':
            model = self.fitStatsModel(X_train, y_train)
            pred = model.predict(X_test.astype(float))
            pred = round(pred)
            stats = pd.DataFrame(classification_report(y_test, pred, output_dict=True))

            return model, stats

        else:
            print("not implemented")
        
    def train(self): 

        # Make data subsets 
        combinations = self.makeCombinations()
        subset_options = self.equalitySubsets(combinations)
        subset_datum = self.makeTestDataSubset(subset_options)
        models = {}
        statistics = {}
        for key in subset_datum:
            subset_x = subset_datum[key][self.data_col]
            subset_y = self.y[self.y.index.isin(subset_x.index)]
            
            if self.library == 'statsmodels':
                temp_model, stats = self.modTest(subset_x, subset_y)
                models[key] = temp_model
                statistics[key] = stats

            if self.library == 'scikit':
                pass


        return models, statistics



