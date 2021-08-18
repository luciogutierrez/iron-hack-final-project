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
df.Race.value_counts()

# Data Visualization
# Demographic distribution
# sns.barplot(data=df, x='Gender', y='Count', hue='Alignment', estimator=sum, palette='hls')
# sns.boxplot(data=df, x='Alignment', y='Power_Rank')

# fig = px.bar(df, x='Gender', y='Count', color='Alignment', barmode='group')
# fig.update_layout(
#     margin=dict(l=20, r=20, t=20, b=20),
#     paper_bgcolor="LightSteelBlue",
# )
# fig.show()

# data = pd.read_csv('./outputs/marvel_data.csv')
# labels = df.Gender.to_list()
# values = df.Count.to_list()
# print(labels, values)
# df = data.groupby(['Race2']).agg({'Count':'sum'}).reset_index()
# df.rename(columns={'Race2':'x', 'Count':'value'}, inplace=True)
# df.to_dict('records')

# new_list = []
# for a_list in race_dict.data:
#     new_dict = dict(a_list)
#     new_list.append(new_dict)
# new_list
# labels = df.Race2
# values = df.Count
# map_lab = list(map(lambda x: "'x'"+':'+"'"+x+"'", labels))
# map_val = list(map(lambda x: "'value'"+':'+str(x), values))
# print(list(map_lab))
# print(list(map_val))
# zip_val = zip(map_lab, map_val)
# tuples_list = list(zip_val)
# tuples_list





# def drop_comilla(s):
#     s = ''.join(ch for ch in s if ch != '"')
#     return s

# new_list = [(drop_comilla(a), drop_comilla(b)) for a, b in tuples_list]
# new_list


# dictionary = df.to_dict('list')
# dictionary
# Power Rank
# data = pd.read_csv('./outputs/marvel_data.csv')
# data.head()
# df = data.groupby(['Race']).agg({'Count':'sum'}).reset_index()
# labels = df.Race.to_list()
# values = df.Count.to_list()
# print(labels, values)

# data.Race.value_counts().sort_values(ascending=False)

# data=[
#     ('01-01-2020',1597),
#     ('02-01-2020',1456),
#     ('03-01-2020',1908),
#     ('04-01-2020',896),
#     ('05-01-2020',755),
#     ('06-01-2020',453),
#     ('07-01-2020',1100),
#     ('08-01-2020',1235),
#     ('09-01-2020',1478)
#         ]
# labels = [row[0] for row in data]
# values = [row[1] for row in data]

# print(labels, values)