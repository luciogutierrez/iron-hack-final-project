# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Data consolidation
# --------------------------------------------------------------------
# data1 = pd.read_csv('./datasets/marvel_character_info.csv')
# data2 = pd.read_csv('./datasets/marvel_powers_matrix.csv')
# data = pd.merge(data1, data2, on=['Name','Alignment'])
# data3 = data[data.Publisher=='Marvel Comics']
# data3.to_csv('./datasets/marvel_data.csv', index=False)

# Data extration
data = pd.read_csv('./datasets/marvel_data.csv')
# --------------------------------------------------------------------

# Data Analysis
# --------------------------------------------------------------------
# data.info()
# data.columns
# data.Alignment.value_counts()

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

def save_df_to_csv(df):
    df.to_csv('./outputs/marvel_data.csv', index=False)

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
    df['Race2'] = np.where(df.Race.isin(race_type), df.Race,'other')
    return df

# Main Pipeline
# --------------------------------------------------------------------
df = selectingVariablesToModel(data)
df = change_variable_name(df, 'Total', 'Power_Rank')
df = adding_count_variable(df)
df = replace_null_values(df, 'Gender', '-', 'undefined')
df = replace_null_values(df, 'Race', '-', 'undefined')
df = make_race_category(df)
save_df_to_csv(df)
# df.info()

# data = pd.read_csv('./outputs/marvel_data.csv')
# table = pd.pivot_table(data, values='Power_Rank', index='Gender', columns='Alignment', aggfunc=np.sum).reset_index()
# headers = list(table.columns.values)
# table = table.fillna(0)
# table.head()