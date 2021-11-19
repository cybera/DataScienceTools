import pandas as pd

def wide_to_long(data_path, data_file, target_col) -> pd.DataFrame:
    
    """
    Format data from wide to long format
    
    Parameters:
    -----------
        data_path (string) full path to data directory
        data_file (string) name of data file (csv format)
        target_col (str) name of column to base reformatting on
    
    """
    
    # Read data into dataframe
    df = pd.read_csv(f'{data_path}{data_file}')
    
    # Reformat
    date_df_long=pd.melt(df,id_vars=['Organization Name','Organization Type'],
                         var_name='Year', 
                         value_name=target_col)
    
    # Remove empty entries
    date_df_long.dropna(subset=[target_col], inplace=True)
    # Reset index
    date_df_long.reset_index(drop=True, inplace=True)
    
    return date_df 

def restack(long_df: pd.DataFrame, data_file: str):
    
    """
    This function takes a long format dataframe and restacks it
    
    Parameters:
    -----------
        long_df obtained after using wide_to_long
        data_file (string) name of data file (csv format)
    
    """
    
    # Rearrange rows
    orgs = long_df['Organization Name'].unique()
    all_dfs = []
    for org in orgs:

        df = long_df[long_df['Organization Name']==org]

        all_dfs.append(df)
        
    final_csv = pd.concat(all_dfs)
    
    final_csv.reset_index(drop=True, inplace=True)
    
    # Save
    outfile_name = data_file.replace("-","- desired")
    final_csv.to_csv(f'{data_path}{outfile_name}')

    
if __name__=="__main__":
    
    # Initialize data variables
    data_path = './data/'
    dates = 'Cybera Membership - join dates.csv'
    fle_fte = 'Cybera Membership - Number of FLE or FTE.csv'

    # Read data
    date_df = pd.read_csv(f'{data_path}{dates}')
    fle_fte_df = pd.read_csv(f'{data_path}{fle_fte}')
    
    # Long format
    long_date_df = wide_to_long(data_path, dates, 'Network Service')
    long_fle_df = wide_to_long(data_path, fle_fte, 'Number of FLE or FTE')
    
    # Restack and save
    restack(long_date_df, dates)
    restack(long_date_df, fle_fte)
