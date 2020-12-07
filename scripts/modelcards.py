import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

class ModelCard:
    '''
    class which will create model score cards to understand how it performs with different subsets of data
    
    This class automatically will find subsets and combindations within a test set and test your model against
    those. Currently, only supports categorical data.
    
    Assumptions: model has a .predict() method (statsmodels and scikit do)
    
    
    known issues as of Nov 20 2020, dfFilter does not work with "weird" column names such as those which contain
    a period (".") or other special characters (/, &, %, etc.) 


    NOTE: At current, this only supports CATEGORICAL models, and true negative/false negative etc. tests
          assume BINARY variables. 
    Variable Documentation:

        model       --> This is any model that has a .predict() method attached to it
        df          --> A data frame of data which will be used for the model and subsetting
        columns     --> The columns at which you want to subset your data to run tests. 
                           NOTE: These columns to not necessarily need to be model features, provided
                           data_col is specified.
        truth       --> a pandas series/numpy vector of the test set truth
        data_col    --> If you want to subset data by variables that are in df, but not model features, 
                        you can pass a list of columns which represent the data that the model expects 
                        to see

        combs       --> If you want to generate combinations of columns to test additional subsets
        comb_size   --> If you have many columns you want to test subests of, you can specify a maximum
                        combination size 
        categorical --> plan for the future, this is always true 

    USAGE:
      To use this we specify our model card:

      card = ModelCard(model = model, 
                       df=df, 
                       columns =[subset_col1, ..., subset_colN],
                       truth = y_test,
                       data_col = list(model_variables),
                       combs = True
                       )

      results = card(as_data_frame = True)


    '''
    def __init__(self, model, df, columns, truth, data_col = None, combs = False, comb_size= None, categorical=True):
        self.model = model
        self.df = df
        self.columns = columns
        self.data_col = data_col
        self.combs = combs
        self.comb_size = comb_size
        self.categorical = categorical
        self.truth = truth
        
    def __accuracyScore(self, subset):
        '''
        Simple function which calculates accuracy of a model 
        '''
        preds = self.__subsetScore(subset.astype(float))
        if self.categorical:
            test_truth = self.truth[self.truth.index.isin(subset.index)]
            # (len(test_truth), len(subset))
            test = np.equal(test_truth.to_numpy().flatten(), preds)
            return len(test[test==True])/len(test)

    def __performanceCount(self, subset):
        '''
        Binary Statistics only for true/false positive 
        '''
        preds = self.__subsetScore(subset.astype(float))

        if self.categorical:
            test_truth = self.truth[self.truth.index.isin(subset.index)]
            CM = confusion_matrix(test_truth, preds).ravel()
            # TN FN FP TP
            return CM.tolist()
   
        
    def __subsetScore(self,subset):
        '''
        assumes the model has a predict method
        '''

        if self.data_col:
            preds = self.model.predict(subset[self.data_col])
        else:
            preds = self.model.predict(subset)
        if self.categorical:
            preds = preds.round()
        
            
        return preds
    
    def __equalitySubsets(self, combs):
        '''
        This function creates a dictionary of each subset combination we're interested in
        and includes the unique values available for those column names
        '''
        values = {}
        if self.combs:
            for comb in combs:
                # if something is alone
                if type(comb) == str:
                    values[comb] = self.df[comb].unique().tolist()
                else:
                    
                    for column in comb:
                        if comb not in values.keys():
                            values[comb] = [self.df[column].unique().tolist()]
                        else:
                            values[comb].append(self.df[column].unique().tolist())

        else: 
            for solo in combs:
                if type(solo) == str:
                    values[solo] = self.df[solo].unique().tolist()
        return values
    
    def dfFilter(self, conds):
        '''
        Function to use a query command in pandas to filter data frames based on 
        strings we buld in makeTestDataSubset
        '''
        cons = [] 
        for cond in conds.split(' & '):
            cons.append(cond.split(' = ' ))
        
        q = ' and '.join(['{0}=={1}'.format(x[0], x[1]) for x in cons])
        
        return self.df.query(q).copy()
    
    def makeTestDataSubset(self, subsets):
        '''
        This function returns a dictionary of filtered data frames representing our filtered
        data for subsets of data we're interested in testing idependently 
        
        '''
        test_dfs = {}
        for key in subsets.keys():
            if type(key) == tuple:
                combinations = list(itertools.product(*subsets[key]))

                for i, comb in enumerate(combinations):
                    bkey = ''
                    for j, c in enumerate(comb):
                        bkey = bkey + str(key[j]) + ' = ' + str(c) + ' & '
                    bkey = bkey.rstrip(' & ')

                    test_dfs[bkey] = self.dfFilter(bkey)
            if type(key) == str:
                for comb in subsets[key]:
                    bkey = str(key) + ' = ' + str(comb) 
                    test_dfs[bkey] = self.dfFilter(bkey)
        return test_dfs

    
    def runTests(self, as_data_frame=False):
        '''
        Currently just runs accuracy tests over the specific subsets that we are 
        interested in
        '''
        scores = {}
        if self.combs:
            if self.comb_size:
                assert self.comb_size < 4, "Too many combinations to search"
                combinations = []

                for i in range(2, comb_size + 1):
                    combinations.extend(list(itertools.combinations(self.columns, i)))
            else:
                assert len(self.columns) < 4, "Too many combinations to search, must specify comb_size"
                combinations = []
                for i in range(1, len(self.columns) + 1):
                    combinations.extend(list(itertools.combinations(self.columns, i)))

            combinations =  combinations
        else: 
            combinations = self.columns

        subset_options = self.__equalitySubsets(combinations)
        subset_datum = self.makeTestDataSubset(subset_options)
        tests = {}
        for key in subset_datum:
            # TN FN FP TP
            performance = self.__performanceCount(subset_datum[key])
            tests[key] = [self.__accuracyScore(subset_datum[key])] + performance + [len(subset_datum[key])]
        
        if as_data_frame:
            tests = pd.DataFrame(tests).T.reset_index()
            cols = ['Subset', 'Accuracy', 'True Negatives', 'False Negatives', 'False Positives', 'True Positives', 'Subset Size']
            tests.columns = cols
                
        return tests
                