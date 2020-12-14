import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from DataSubsetter import DataSubsetter
import numpy as np

# For continuous tests, does it make sense to do some binning? 

class ModelCard(DataSubsetter):
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

        combs       --> If you want to generate combinations of columns to test additional subsets. 
                        If set to False, will simply iterate through possible values of each column in
                        columns and return those results in isolation. 

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
    def __init__(self, df, columns, model, truth,comb_size = None, data_col = None, combs = False, categorical=True, q=4):
        DataSubsetter.__init__(self, df, columns, comb_size, q)
        self.model = model
        self.data_col = data_col
        self.combs = combs
        self.categorical = categorical
        self.truth = truth
        self.test_cols = None
        self.q = q
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
        # TODO: Will need to update this to handle data which is categorical, not binary
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
    

    def comboFlag(self, quantile = 0.75, metric = 'Accuracy', direction = 'positive'):
        '''
        Funciton to automatically filter results to things we might be worried about. 

        quantile is the data quantile of the metric that data should be greater or less than,
        metric is a column of the output of runTests, and the direciton positive is for 
        things that perform better than quantile, negative isf or things that perform worse
        '''

        try:
            self.tests.head()
        except AttributeError:
            self.runTests()
        if isinstance(self.tests, dict):
            self.tests = pd.DataFrame(self.tests, columns = self.test_cols)

        if direction == 'positive':
             warning_df = self.tests[self.tests[metric] >=  self.tests[metric].quantile(quantile)]
        elif direction == 'negative':
             warning_df = self.tests[self.tests[metric] <=  self.tests[metric].quantile(quantile)]
        else:
            print('{direction} is not a valid choice for direction, only accepts "positive" or "negative".')
            return 
        return warning_df
        

    def runTests(self, as_data_frame=True, counts = False):
        '''
        Currently just runs accuracy tests over the specific subsets that we are 
        interested in
        '''
        scores = {}

        subset_datum = self.makeTestDataSubset()
        tests = {}
        for key in subset_datum:
            # TN FN FP TP
            # TODO: Will need to update this to handle data which is categorical, not binary
            p = self.__performanceCount(subset_datum[key])
            
            precision = p[3] / (p[3] + p[2])
            recall = p[3] / (p[3] + p[1])
            try:
                F1 = 2 * (precision * recall) / (precision + recall)
            except: 
                F1 = np.nan
            tests[key] = [self.__accuracyScore(subset_datum[key])] 
            tests[key] += [precision, recall, F1]
            if counts:
                tests[key] += p
        
            tests[key] += [len(subset_datum[key])]
        if as_data_frame:
            cols  = ['Subset', 'Accuracy', 'Precision', 'Recall', 'F1']
            if counts:
                cols += ['True Negatives', 'False Negatives', 'False Positives', 'True Positives']
            cols += ['Subset Size']
            tests = pd.DataFrame(tests).T.reset_index()
            tests.columns = cols
        self.test_cols = cols
        self.tests = tests           
        return tests
                