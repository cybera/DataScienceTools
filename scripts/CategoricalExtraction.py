import pandas as pd 
from sklearn import preprocessing
import numpy as np

class DummyExtract():
    '''
    Python class to take pandas dataframes with columns of encodings, for example if you
    have a column that looks like

        DataClasses
        1;2;3
        6;8
        NaN
        1
        ...
    
    It is more useful to convert those into a sparse one-hot encoded dataframe where we have the same 
    number of rows, but a binary flag for each integer in that row instead. In this case, we will have 
    N columns, where N is the maximum integer flag in the above classes.
    '''

    def __init__(self, df, col=None, kind='onehot', codes = None, 
                 sep = ';', minus = 1, join=True, keys="ULI",
                  min_separators = 0, keep_nan = True, verbose = True, 
                  filter_cols = False,
                  min_not_na = 100, test_bin = True, bin_test_dropna = True):
        self.col = col
        self.codes = codes
        self.df = df.copy()
        self.sep = sep
        self.minus = minus
        self.join = join
        self.keys = keys
        self.dummies = None
        self.kind = kind
        self.labels = None
        self.inverse = None
        self.min_separators = min_separators
        self.keep_nan = keep_nan
        self.verbose = verbose
        self.filter_cols = filter_cols
        self.min_not_na = min_not_na
        self.test_bin = test_bin
        self.bin_test_dropna = bin_test_dropna
    
    def filler(self, row):
        '''
        This function will fill in a table of values
        from the list or series codes from the integer
        codes by default. This will fill in the actual 
        word as opposed to simply the integer value. Useful
        for human readability
        '''
        conds = row[self.col]
        fill = []
        for idx in conds:
            try:
                idx = int(idx)
            except ValueError:
                # nan values
                continue
            # need to subtract because data base is 1 indexed, and pandas is 0 indexed
            fill.extend([self.codes[idx - self.minus]])
        return fill

    

    def extractDummySeries(self):
        '''
        This function extracts the series of interest for dummy encoding. This will create
        a series of entries like 
        
            0 : [keyword1, .... , keywordN]
            1 : [keyword1, .... , keywordN]
        '''
        # if not self.keep_nan:
        #     self.df[self.col] = self.df[self.col].apply(lambda x: str(x).split(self.sep) if not pd.isnull(x) else [None] )
        # else:
        self.df[self.col] = self.df[self.col].apply(lambda x: [x.strip() for x in str(x).split(self.sep)])
        
        if self.codes is None:
            return self.df[self.col]
    
        self.df[self.col] = self.df.apply(self.filler, axis=1)
        return self.df[self.col]

    def extractEncodingSeries(self, fill = "None"):
        '''
        In this case, if we don't have multiple choices, one-hot-encoding isn't ideal
        so we present a method which allows us to create integer labels for different
        things.
        '''
        le = preprocessing.LabelEncoder() 
        
        # Empty can be an important class as well 
        
        self.labels = self.df[self.col].fillna(fill)
    
        le.fit(self.labels)

        self.labels = pd.DataFrame(le.transform(self.labels))
        self.labels.columns = [self.col + "_labels"]
        # hold on to the inverse transform for when we need it
        self.inverse = pd.DataFrame(pd.Series(le.classes_)).reset_index()
        self.inverse.columns = ['encoding' , 'value']

        return self.labels


    def codeDummies(self, series):
        '''
        This takes the series from extractDummySeries, and turns those keys into a 
        sparse dummy pandas data frame
        '''
        if self.kind == 'onehot':
            s_frame = series.apply(pd.Series).stack()
            s_frame = s_frame.replace('nan', np.nan)
            self.dummies = pd.get_dummies(s_frame, dummy_na=self.keep_nan).sum(level=0)
            self.dummies.columns = [str(i) for i in self.dummies.columns]

    
    
    def __dropSparse(self):
        '''
        Little internal function which will ignore columns that are mostly empty; 
        no need to encode them if there's only a few non na values
        '''
        counter = self.df[self.col].value_counts()
        count = counter.sum()
        if count < self.min_not_na:
            print("Dropping", self.col, 'with non-na value count of', count)
            self.col = []
            return
        elif len(counter.index) <= 1:
            print("Dropping", self.col, "Has only a single variable type" )
            self.col = []
            return

    def testBinaryCategorical(self):
        '''
        sometimes a column is technically a binary column and making two
        sparse vectors would be dumb
        '''
        if self.bin_test_dropna:
            if len(self.df[self.col].dropna().unique()) == 2:
                return True
        
        if len(self.df[self.col].unique()) == 2:
            return True
        
        else:
            return False

    def extract(self):
        '''
        convenience function to combine the above methodologies. 
        if we want, it will join with patient identifiers and automatically 
        adapt to if there are codes to swap if we feel like it 
        '''
        if self.filter_cols:
            self.__dropSparse()
            if len(self.col) == 0:
                return 
        if self.test_bin:
            if self.testBinaryCategorical():
                print(self.col, 'Was binary')
                self.kind = 'labels'
                internal = self.extractEncodingSeries(fill='0')
                if self.join:
                    self.labels = self.labels.join(self.df[self.keys])
                    return

        if self.kind == 'onehot':
            internal = self.extractDummySeries()
            self.codeDummies(internal)

            if self.join:
                self.dummies = self.dummies.join(self.df[self.keys])
        
        if self.kind == 'labels':
            internal = self.extractEncodingSeries()
            if self.join:
                self.labels = self.labels.join(self.df[self.keys])

    def smartExtract(self):
        '''
        Different convenience function which will do it's best to figure out 
        what kind of encoding is best to return. 
        '''
        sep_count = self.df[self.col].astype(str).str.contains(self.sep).sum()
        # assume we need one-hot-encoding if any entries have more than min_separators
        # of separators
        if sep_count > 100:
            print(self.col, sep_count)
        if sep_count > self.min_separators:
            self.kind = 'onehot'
            self.extract()
        else:
            self.kind = 'labels'
            self.extract()

    def returnFrame(self, min_sum=0):
        '''
        spits out the data frame
        '''
        if len(self.col) == 0:
            return None, None
        
        if self.kind == 'labels':
            return self.labels, self.inverse
        
        if self.dummies is None:
            print("need to extract dummies")
            return
        
        elif len(self.col) !=0:
            return self.dummies[self.dummies.columns[self.dummies.sum() > min_sum]], None

class ExtractAll(DummyExtract): 
    '''
    DummyExtract works on a pandas series, this class is the implementation
    that will extract categorical variables for an entire data frame, and 
    return a dictionary of data frames, or by default, a new dataframe suitable
    for ingestion
    '''
    
    def __init__(self, features, smart = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = features
        self.frames = {}
        self.smart = smart

    def masterJoin(self, master_min_sum = 0):
        '''
        Function to join all our data frames into one, better data frame
        '''
        dict_keys = list(self.frames.keys())
        out_frame = self.frames[dict_keys[0]]
        for key in dict_keys[1:]:
            print(len(out_frame), len(self.frames[key]), key, dict_keys[0])
            out_frame = pd.merge(out_frame, self.frames[key], on = self.keys, how = 'left')
            key1 = key
        return out_frame[out_frame.columns[out_frame.sum() > master_min_sum]]
        
    def extractAll(self, master_min_sum = 0, min_sum = 0, master_join = False):
        '''
        wrapper for DummyExtract to work its way through each column of a data frame
        '''
        # The word "None" is going to be an issue here 
        
        if (master_join) and (self.frames !={}):
            return self.masterJoin(master_min_sum = master_min_sum)
        
        if self.frames != {}:
            return self.frames

        self.df[self.keys] = self.df[self.keys].astype(str)
        for col in self.features:
            extract = DummyExtract(df=self.df, col= col, kind=self.kind,
                                    codes = self.codes, 
                                    sep = self.sep, minus = self.minus, 
                                    join=self.join, keys=self.keys,
                                    min_separators = self.min_separators, 
                                    keep_nan = self.keep_nan, verbose = self.verbose, 
                                    filter_cols = self.filter_cols,
                                    min_not_na = self.min_not_na) 
            # Probably need a flag for smart extraction 
            if self.smart:
                extract.smartExtract()
            else:
                extract.extract()
            frame = extract.returnFrame(min_sum = min_sum)[0]
            
            if isinstance(frame, pd.DataFrame):
                frame.columns = [col + "_" + c if c != self.keys else self.keys for c in frame.columns]
                self.frames[col] = frame

        if master_join:
            return self.masterJoin(master_min_sum = master_min_sum)

        return self.frames