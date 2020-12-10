import pandas as pd 
import numpy as np 
from DataSubsetter import DataSubsetter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV


class SubsetModelTrainer(DataSubsetter):
    # This might be a sign i need to do more abstraction....
    def __init__(self, df, y, columns, model, comb_size = None, 
                 data_col = None, combs=True, library='statsmodels', 
                 kwargs={}, train_test_split = False, ttprop = 0.2, fit_type = 'fit',
                 stats_to_df = True, drop_subset_column = True, feature_select = False,
                 feature_selection = None, folds = 3, score='accuracy'):
        
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
        self.stats_to_df = stats_to_df
        self.drop_subset_column = drop_subset_column
        self.feature_select = feature_select
        self.feature_selection = feature_selection
        self.folds = folds
        self.score = score

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

    def internalFeatureSelect(self, X, y):
        '''
        Recursive K-fold feature selection)
        '''
        setup = self.model(**self.kwargs)
        rfecv = RFECV(estimator = setup, 
                      cv = StratifiedKFold(self.folds), 
                      scoring = self.score, n_jobs=-1)
        rfecv.fit(X, y)

        features = rfecv.support_
        if self.drop_subset_column:
            internal_data_col =  [i for i in self.data_col if i not in self.columns]
        else:
            internal_data_col = self.data_col
        features = [internal_data_col[i] for i in range(len(internal_data_col)) if features[i]]
        
        setup.fit(X[features], y)
        
        return setup, features


    def modTest(self, x, y):
        if self.train_test_split:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.ttprop)
        else: 
            X_train = X_test = x
            y_train = y_test = y
        
        if self.library == 'statsmodels':
            if self.feature_select:
                raise NotImplementedError("Feature selection with stats models not implemented" )
            model = self.fitStatsModel(X_train, y_train)
       
        elif self.library == 'scikit':
            if self.feature_select:
                if self.feature_selection:
                    raise NotImplementedError("Custom Functions not currently supported")
                else: 
                    model, features = self.internalFeatureSelect(X_train, y_train)
            else:
                model = self.fitSciKit(X_train, y_train)
        
        else:
            msg = self.library + " is not not implemented"
            raise NotImplementedError(msg)
            
        if self.feature_select:
            X_test = X_test[features]
        pred = pd.Series(model.predict(X_test.astype(float)))
        pred = round(pred)
        stats = pd.DataFrame(classification_report(y_test, pred, output_dict=True))

        if self.feature_select:
            return model, stats, features
        return model, stats
    
    def train(self): 

        # Make data subsets 
        combinations = self.makeCombinations()
        subset_options = self.equalitySubsets(combinations)
        subset_datum = self.makeTestDataSubset(subset_options)
        models = {}
        statistics = {}
        features = {}
        for key in subset_datum:
            print("Training subset: ", key)
            subset_x = subset_datum[key][self.data_col]
            subset_y = self.y[self.y.index.isin(subset_x.index)]
            
            if self.drop_subset_column:
                # As everything is now a single value, we need
                # to drop these columns to avoid singular matricies
                subset_x = subset_x.drop(self.columns, axis = 1)
                
            if self.feature_select:    
                temp_model, stats, feature = self.modTest(subset_x, subset_y)
                features[key] = feature
            else: 
                temp_model, stats = self.modTest(subset_x, subset_y)
            models[key] = temp_model
            statistics[key] = stats

        # convert to easy to read DF
        if self.stats_to_df:
            statistics = pd.concat({k: pd.DataFrame(v) for k, v in statistics.items()})
        if self.feature_select:  
            return models, statistics, features
        
        return models, statistics



