# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Extraction of data
data1 = pd.read_csv('./datasets/marvel_character_info.csv')
data2 = pd.read_csv('./datasets/marvel_powers_matrix.csv')
data = pd.merge(data1, data2, on=['Name','Alignment'])
# data.to_csv('./outputs/marvel_data.csv', index=False)

# Analysis of data
data.info()
data.columns
data.Alignment.value_counts()

# Transformation of data
# Select variables to include in the model
# --------------------------------------------------------------------
def selectingVariablesToModel(df):
    """
      ['ID', 'Name', 'Alignment', 'Gender', 'EyeColor', 'Race', 'HairColor',
       'Publisher', 'SkinColor', 'Height', 'Weight', 'Intelligence',
       'Strength', 'Speed', 'Durability', 'Power', 'Combat', 'Total']    
    """
    df = df.copy()
    model_cols = ['Name','Alignment','Gender','Race','Publisher','Height','Weight', 
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

# Main Pipeline
# --------------------------------------------------------------------
df = selectingVariablesToModel(data)
df = change_variable_name(df, 'Total', 'Power_Rank')
df = adding_count_variable(df)
df = replace_null_values(df, 'Gender', '-', 'undefined')
save_df_to_csv(df)
df.info()
df.head()
df.Gender.value_counts()

data = pd.read_csv('./outputs/marvel_data.csv')
df = data.groupby(['Gender']).agg({'Count':'sum'}).reset_index()
labels = df.Gender.to_list()
values = df.Count.to_list()
print(labels, values)

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



# Data Visualization
# Demographic distribution
sns.barplot(data=df, x='Gender', y='Count', hue='Alignment', estimator=sum, palette='hls')

fig = px.bar(df, x='Gender', y='Count', color='Alignment', barmode='group')
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",
)
fig.show()

# Power Rank
sns.boxplot(data=df, x='Alignment', y='Power_Rank')