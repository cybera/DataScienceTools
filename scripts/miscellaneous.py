# misc funcs

from itertools import combinations

def comboFinder(sets, appear_once = False):
    '''
    Function which takes a dictionary of sets which contain the set intersection
    of various combinations of those features. For example, if we have three sets 
    S1, S2 and S3 as input, this will return the following combinations of overlap,
    as well as the "leftovers" unique to each set. Mathematically speaking: 
    
    S1 ∩ S2 ∩ S3
    S1 ∩ S2
    S2 ∩ S3
    S1 ∩ S3
    
    S1 - (the above intersections)
    S2 - (the above intersections)
    S3 - (the above intersection)
    
    Useful for understanding how different sets are related. 
    
    INPUT:
    
    sets  -->  Dictionary of sets
    appear_once --> Flag to filter additional overlap. If true, each element will only appear once
                    accros all output sets. If False, elements of each list may appear in more than 
                    one place. 

    RETURNS:

        dictionary, keyed by tuples of the keys of sets with the intersection between those sets, filtered
        if flag is present 
    
    '''
    combo_uniques = {}
    temp = {}
    visited = []
    for key in sets:
        temp[(key,)] = sets[key]
    
    for i in range(len(sets) , 1, -1):
        comb = list(combinations(sets, i))
        mini_set = set()
        
        for j, tup in enumerate(comb):
            for i, key in enumerate(tup):
                if i == 0:
                    mini_set = sets[key]
                else:
                    mini_set =  (mini_set & sets[key])
             # remove overlap   
            if appear_once:
                for key in combo_uniques:
                    mini_set = mini_set - combo_uniques[key]
            
            combo_uniques[tup] = mini_set
       
    
    for tup in combo_uniques.keys():
        for key in tup:
            temp[(key,)] = temp[(key,)] - combo_uniques[tup]

           
    
    combo_uniques.update(temp)
    return combo_uniques