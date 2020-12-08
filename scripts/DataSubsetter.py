
import itertools
import pandas as pd
import numpy as np

class DataSubsetter:
    '''
    Class to create data subsets based on combinations of data
    '''
    def __init__(self, df, columns, comb_size = None):
        self.df = df 
        self.columns = columns
        self.comb_size = comb_size
    
    def equalitySubsets(self, combs):
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

    def makeCombinations(self):

        if self.comb_size:
            assert self.comb_size <= 4, "Too many combinations to search"
            combinations = []

            for i in range(1, self.comb_size + 1):
                combinations.extend(list(itertools.combinations(self.columns, i)))
        else:
            assert len(self.columns) <= 4, "Too many combinations to search, must specify comb_size"
            combinations = []
            for i in range(1, len(self.columns) + 1):
                combinations.extend(list(itertools.combinations(self.columns, i)))

        return combinations
    
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
