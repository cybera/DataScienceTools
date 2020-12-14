import pandas as pd 
import numpy as np 
from DataSubsetter import DataSubsetter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class SubsetModelTrainer(DataSubsetter):
    
    def __init__(self, df, y, columns, model, comb_size = None, 
                 data_col = None, combs=True, library='scikit', 
                 kwargs={}, train_test_split = False, ttprop = 0.2, fit_type = 'fit',
                 stats_to_df = True, drop_subset_column = True, q=4):
        
        DataSubsetter.__init__(self, df, columns, comb_size,  q)
        self.model = model
        self.data_col = data_col
        self.combs = combs
        self.kwargs = kwargs
        self.y = y
        self.train_test_split = train_test_split
        self.ttprop = ttprop
        self.library = library
        self.fit_type = fit_type
        self.stats_to_df = stats_to_df
        self.drop_subset_column = drop_subset_column
        

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
        print('     Number of rows in training set:', len(X_train))
        if self.library == 'statsmodels':
            model = self.fitStatsModel(X_train, y_train)
       
        elif self.library == 'scikit':
            model = self.fitSciKit(X_train, y_train)
        
        else:
            print(self.library, " is not not implemented")
            raise NotImplementedError
            
        pred = pd.Series(model.predict(X_test.astype(float)))
        pred = round(pred)
        stats = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        return model, stats
    
    def train(self): 

        # Make data subsets 
        subset_datum = self.makeTestDataSubset()
        models = {}
        statistics = {}
        for self.key in subset_datum:
            print("Training subset: ", self.key)
            subset_x = subset_datum[self.key][self.data_col]
            subset_y = self.y[self.y.index.isin(subset_x.index)]
            
            if self.drop_subset_column:
                # As everything is now a single value, we need
                # to drop these columns to avoid singular matricies
                drop_cols = []
                for col in self.columns:
                    if self.typeCheck(self.df[col]) == 'int':
                        drop_cols.append(col)
                        print('     Removing filter column', col, "from model")
        
                subset_x = subset_x.drop(drop_cols, axis = 1)
                
            temp_model, stats = self.modTest(subset_x, subset_y)
            models[self.key] = temp_model
            statistics[self.key] = stats

        # convert to easy to read DF
        self.models = models
        if self.stats_to_df:
            statistics = pd.concat({k: pd.DataFrame(v) for k, v in statistics.items()})
        return models, statistics