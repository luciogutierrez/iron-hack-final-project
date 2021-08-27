# imports
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Data extration
this_folder = os.path.dirname(os.path.abspath(__file__))
df_train = pd.read_csv(this_folder + '/datasets/marvel_data.csv')
df_target = pd.read_csv(this_folder + '/datasets/dc_data.csv')
# --------------------------------------------------------------------

# Data Analysis
# --------------------------------------------------------------------
def percentage_nulls(df):
    number_nulls = pd.DataFrame(df.isnull().sum(),columns=['Total'])
    number_nulls['% nulls'] = round((number_nulls['Total'] / df.shape[0])*100,1)
    return number_nulls

def display_all(df):
    with pd.option_context("display.max_rows",1000 ,  "display.max_columns", 1000): 
        display(df)
        
def analysis_train_set():
    df_train.head()
    df_train.info()
    df_train.isna().sum()
    percentage_nulls(df_train)
    df_train.Alignment.value_counts()
    df_train.Alignment.value_counts(normalize=True)
    df_train.Gender.value_counts()
    df_train.columns
    display_all(df_train.describe(include='all').T)
    return

def analysis_target_set():
    df_target.head()
    df_target.info()
    df_target.isna().sum()
    percentage_nulls(df_target)
    df_target.Alignment.value_counts()
    df_target.Alignment.value_counts(normalize=True)
    df_target.Gender.value_counts()
    df_target.columns
    display_all(df_target.describe(include='all').T)
    return

# Data Transformation
# Select variables to include in the model
# --------------------------------------------------------------------
def selectingVariablesToModel(df):
    """
      ['ID', 'Name', 'Alignment', 'Gender', 'EyeColor', 'Race', 'HairColor',
       'Publisher', 'SkinColor', 'Height', 'Weight', 'Intelligence',
       'Strength', 'Speed', 'Durability', 'Power', 'Combat', 'Total']    
    """
    df = df.copy()
    model_cols = ['Name','Alignment','Gender','Race','Height','Weight', 
                  'Intelligence','Strength','Speed','Durability','Power','Combat','Total']
    df = df[model_cols]
    return df

# Change variable name
# --------------------------------------------------------------------
def change_variable_name(df, old_name, new_name):
    df = df.copy()
    df.rename(columns={old_name:new_name}, inplace=True)
    return df

# Change variable name
# --------------------------------------------------------------------
def adding_count_variable(df):
    df = df.copy()
    df['Count'] = 1
    return df

# Change variable name
# --------------------------------------------------------------------
def replace_null_values(df, col_name, old_value, new_value):
    df = df.copy()
    df.loc[df[col_name] == old_value, col_name] = new_value
    return df

def save_df_to_csv(df, studios):
    if studios=='Marvel':
        df.to_csv(this_folder + '/outputs/marvel_data.csv', index=False)
    else:    
        df.to_csv(this_folder + '/outputs/dc_data.csv', index=False)

def make_race_category(df):
    race_type = ['undefined',
                'Human',
                'Mutant',
                'Human / Radiation',
                'Symbiote',
                'Asgardian',
                'Demon',
                'God / Eternal',
                'Alien',
                'Inhuman',
                'Android',
                'Human / Cosmic',
                'Eternal',
                'Cosmic Entity'
                'Cyborg',
                'Human / Altered',
                'Alpha']
    df = df.copy()
    # df['Race2'] = np.where(df.Race.isin(race_type),df.Race,'other')
    # df['Race2'] = np.where(~df['Race'].isin(race_type),'other', df['Race'])
    df['Race'] = np.where(df.Race.isin(race_type), df.Race,'other')
    return df

# Main Pipeline
# --------------------------------------------------------------------
# data preparation for marvel model
df = selectingVariablesToModel(df_train)
df = change_variable_name(df, 'Total', 'Power_Rank')
df = adding_count_variable(df)
df = replace_null_values(df, 'Gender', '-', 'undefined')
df = replace_null_values(df, 'Race', '-', 'undefined')
df = make_race_category(df)
save_df_to_csv(df,'Marvel')

# data preparation for marvel model
df = selectingVariablesToModel(df_target)
df = change_variable_name(df, 'Total', 'Power_Rank')
df = adding_count_variable(df)
df = replace_null_values(df, 'Gender', '-', 'undefined')
df = replace_null_values(df, 'Race', '-', 'undefined')
df = make_race_category(df)
save_df_to_csv(df,'Dc')