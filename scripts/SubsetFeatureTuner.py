import pandas as pd 
import numpy as np 
from SubsetModels_nofeat import SubsetModelTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV


class SubsetFeatureTuner(SubsetModelTrainer):

    def __init__(self,df, y, columns, search, search_params = {}, 
                 train_test_split = True,ttprop=0.2, kwargs={},
                 search_type = "hyperparameter"):
        SubsetModelTrainer.__init__(self,df=df, y=y, 
                                    columns=columns, model=search_params['estimator'],
                                    train_test_split = train_test_split, ttprop=ttprop, kwargs = kwargs)
        self.search_params = search_params
        self.search = search
        self.features = {}
        self.kwargs = {}
        self.search_type = search_type
    
    def paramSearch(self, X, y):
        int_search_dict = self.search_params.copy()
        int_search_dict['estimator'] = int_search_dict['estimator']()
        searcher = self.search(**int_search_dict)
        searcher.fit(X, y)

        return searcher 

    def featureSelection(self, X, y):
        '''
        Recursive K-fold feature selection)
        '''
        setup = self.model(**self.kwargs)
        rfecv = RFECV(estimator = setup, 
                      cv = StratifiedKFold(n_splits=3, shuffle=True), 
                      scoring = 'f1', n_jobs=-1, min_features_to_select = 5)
        rfecv.fit(X, y)
        self.features[self.key] = rfecv.support_
        return rfecv

    def getHyperParams(self, key):
        try:
            return self.models[key].best_params_
        except AttributeError:
            return self.kwargs

    def getModelFeatures(self, key):
        try:
            self.features[key]
        except KeyError:
            print(key, "Not in features")
            return
        if self.drop_subset_column:
            cols = [x for x in self.data_col if x not in self.columns]
        else:
            cols = self.data_col
        kept_features = [cols[i] for i in range(len(cols)) if self.features[key][i]]
        return kept_features

    def fitFeatureModel(self, X, y, key):
        model  = self.model(**kwargs)
        X = X[self.getFeatures[key]]
        model.fit(X, y)
        
        return model 

    def __pred(self, model, X):
        try:
            return pd.Series(model.best_estimator_.predict(X))
        except AttributeError:
            return pd.Series(model.predict(X))
    
    def modTest(self, x, y):
        if self.train_test_split:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.ttprop)
        else: 
            X_train = X_test = x
            y_train = y_test = y
        
        if self.library == 'scikit':
            if self.search_type == 'hyperparameter':
                model = self.paramSearch(X_train, y_train)
            elif self.search_type == 'feature':
                model = self.featureSelection(X_train, y_train)
            elif self.search_type == 'both':
                raise NotImplementedError("Simultaneous optimization not currently supported")
            else:
                raise NotImplementedError(self.search_type, "is not implemented")
       
        else:
            print(self.library, " is not not implemented for model tuning")
            raise NotImplementedError
            

        pred = self.__pred(model, X_test.astype(float))
        pred = round(pred)
        stats = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        return model, stats 








        


