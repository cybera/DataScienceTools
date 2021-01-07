
import itertools
import pandas as pd
import numpy as np
'''
known issues as of Nov 20 2020, dfFilter does not work with "weird" column names such as those which contain
    a period (".") or other special characters (/, &, %, etc.) 
    '''
class DataSubsetter:
    '''
    Class to create data subsets based on combinations of data
    '''
    def __init__(self, df, columns, comb_size = None, q = 4, static=False):
        self.df = df 
        self.columns = columns
        self.comb_size = comb_size
        self.q = q
        self.static = static
    
    def typeCheck(self, column):
        # Check for int
        ints = column.astype(str).str.isdigit().all()
        if ints:
            return 'int'
        # check for floats
        floats = (True if 'float' in [column.dtypes] else False)
        if floats:
            return 'float'
        # boolean okay too, treated as integer case 
        if column.dtypes == 'bool':
            return 'int'
        else:
            raise TypeError('Column "{}" has type "{}" which cannot be subsetted'.format(column.name, column.dtype))


    def continuousSubset(self, column):
        intervals = pd.qcut(self.df[column], q = self.q, duplicates='drop').unique().tolist()
        query_list = []
        for inter in intervals:
            query_list.extend([{'col':column, 
                                'left':str(inter.left), 
                                'right':str(inter.right)}])
            
        return query_list
    
    def makeSubsets(self, combs):
        '''
        This function creates a dictionary of each subset combination we're interested in
        and includes the unique values available for those column names
        '''

        values = {}
        for comb in combs:
            # if something is alone
            if type(comb) == str:
                type_ = self.typeCheck(self.df[comb])
                if type_ == 'int':
                    values[comb] = self.df[comb].unique().tolist()
                elif type_ == 'float':
                    values[comb] = self.continuousSubset(comb)
            else:
                for column in comb:
                    
                    type_ = self.typeCheck(self.df[column])
                    if comb not in values.keys():
                        if type_ == 'int':
                            values[comb] = [self.df[column].unique().tolist()]
                        elif type_ == 'float':
                            values[comb] = [self.continuousSubset(column)]
                    else:
                        if type_ == 'int':
                            values[comb].append(self.df[column].unique().tolist())
                        elif type_ == 'float':
                            values[comb].append(self.continuousSubset(column))

        return values

    def makeCombinations(self):

        if self.comb_size:
            assert self.comb_size <= 4, "Too many combinations to search"
            combinations = []
            if not self.static:
                for i in range(1, self.comb_size + 1):
                    combinations.extend(list(itertools.combinations(self.columns, i)))
            else:
                combinations.extend(list(itertools.combinations(self.columns, 2)))
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

        return self.df.query(conds).copy()
    
    def __queryMake(self, dic):
        string = dic['left'] + ' < ' + dic['col'] + ' & ' + dic['col'] + " < " + dic['right']
        return string
   
    def __makeQuery(self, comb, key, friends = True):
        query_key = ''
        if friends:
            for j, c in enumerate(comb):
                
                if self.typeCheck(self.df[key[j]]) == 'int':
                    query_key = query_key + str(key[j]) + ' == ' + str(c) + ' & '
                
                elif self.typeCheck(self.df[key[j]]) == 'float':
                    query_key = query_key + self.__queryMake(c) + ' & '
        else:
            pass


        return query_key.strip(' & ')
    
    def makeTestDataSubset(self):
        '''
        This function returns a dictionary of filtered data frames representing our filtered
        data for subsets of data we're interested in testing idependently 
        
        '''
        combinations = self.makeCombinations()
        subsets = self.makeSubsets(combinations)
        
        test_dfs = {}
        for key in subsets.keys():
            if type(key) == tuple:
                combinations = list(itertools.product(*subsets[key]))
                  
                for i, comb in enumerate(combinations):
                   
                    bkey = self.__makeQuery(comb, key)
                    print(bkey)
                    test_dfs[bkey] = self.dfFilter(bkey)
            if type(key) == str:
                for comb in subsets[key]:
                  
                    bkey = str(key) + ' = ' + str(comb) 
                    
                    test_dfs[bkey] = self.dfFilter(bkey)
        return test_dfs
