# --------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# --------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------
this_folder = os.path.dirname(os.path.abspath(__file__))
df_train = pd.read_csv(this_folder + '/outputs/marvel_data.csv')
df_target = pd.read_csv(this_folder + '/outputs/dc_data.csv')

# --------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------
# Train set
# --------------------------------------------------------------------
def analysis_df_train():
    df_train.head(5)
    df_train.info()
    df_train.describe()
    df_train.isna().sum()
    df_train.Race.value_counts()
    df_train.Alignment.value_counts()
    df_train.Alignment.value_counts(normalize=True)
    df_train.Gender.value_counts()
    df_train.Gender.value_counts(normalize=True)
    df_train.columns
    corr = df_train.drop('Power_Rank', axis=1).corr()
    sns.heatmap(corr, annot=True, cmap='RdYlGn')
    return

# Target set
# --------------------------------------------------------------------
def analysis_df_target():
    df_target.head(5)
    df_target.info()
    df_target.describe()
    df_target.isna().sum()
    df_target.Race.value_counts()
    df_target.Alignment.value_counts()
    df_target.Alignment.value_counts(normalize=True)
    df_target.Gender.value_counts()
    df_target.Gender.value_counts(normalize=True)
    df_target.columns
    corr = df_target.drop('Power_Rank', axis=1).corr()
    sns.heatmap(corr, annot=True, cmap='RdYlGn')
    return


# --------------------------------------------------------------------
# Transformation
# --------------------------------------------------------------------

# Drop catecorical values that are not present in the train set
# --------------------------------------------------------------------
def depuraTrainSet(df, df2):
    df = df.copy()
    # cambia por "indefinido" la raza para personajes de razas que no existen en el set objetivo
    df2_list = df2.Race.value_counts().index.tolist()
    # df.drop(df[~df.Race.isin(df2_list)].index, inplace = True)
    df.Race = np.where(df.Race.isin(df2_list), df.Race, 'undefined')
    
    # depura del set de entrenamiento personajes sin genero definido ya no existen de este tipo en el set objetivo
    df.drop(df[df.Gender=='undefined'].index, axis=0, inplace=True)
    return df

# Drop outliers
# --------------------------------------------------------------------
def dropOutliers(df):
    # identificamos las columnas númericas y las categoricas
    n_cols = ['training_hours']
    # Eliminamos los outliers de las columnas númericas
    z_scores = stats.zscore(df[n_cols])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df = df[filtered_entries]
    df.describe()
    return df

# Drop null values
# --------------------------------------------------------------------
def drop_null_values(df):
    df = df.copy()
    df.dropna(inplace=True)
    return df

# Drop extra values for dependet variable
# (for the purpose of the model we dont' need it)
# --------------------------------------------------------------------
def drop_neutral_values(df):
    df = df.copy()
    df.drop(df[df.Alignment=='neutral'].index, axis=0, inplace=True)
    return df

# Selecting variables to include in the model
# --------------------------------------------------------------------
def selectingVariablesToModel(df):
    """
    ['Name', 'Alignment', 'Gender', 'Race', 'Height', 'Weight',
       'Intelligence', 'Strength', 'Speed', 'Durability', 'Power', 'Combat',
       'Power_Rank', 'Count']
    """
    model_cols = ['Name', 'Alignment', 'Gender', 'Race', 'Height', 'Weight',
       'Intelligence', 'Strength', 'Speed', 'Durability', 'Power', 'Combat']
    df_cols = df[model_cols]
    return df_cols

# Change Alignment variale to numbers
# --------------------------------------------------------------------
def changeToNumberDependientVariable(df):
    df = df.copy()
    df.Alignment = np.where(df.Alignment=='good',1,0)
    return df

# Change type of variables that should be categorical
# --------------------------------------------------------------------
def changeTypeToCagory(df, cols):
    df = df.copy()
    for col in cols:
        df[col] = df[col].astype('category')
    return df

# balance train set with over-sampling
# --------------------------------------------------------------------
def balanceDownSampling(df):
    df = df.copy()
    df_1 = df[df.Alignment == 1]
    df_0 = df[df.Alignment == 0]
    df_resample = resample(df_1, replace=True, n_samples=df_0.shape[0])
    df_bal = pd.concat([df_resample, df_0])
    df_bal = df_bal.sample(frac=1)
    return df_bal

# balance train set with over-sampling
# --------------------------------------------------------------------
def balanceOverSampling(df):
    df = df.copy()
    df_1 = df[df.Alignment == 'good']
    df_0 = df[df.Alignment == 'bad']
    df_resample = resample(df_0, replace=True, n_samples=df_1.shape[0])
    df_bal = pd.concat([df_resample, df_1])
    df_bal = df_bal.sample(frac=1)
    return df_bal

# Scale numerical variables
# --------------------------------------------------------------------
def ScaleNumericVariables(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['float','int64']).columns.tolist()
    scaler = StandardScaler()
    scaler.fit(df[numeric_cols])
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df

# Generate dummy variables
# --------------------------------------------------------------------
def generateDummySet(df):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

# Train Pipeline
# --------------------------------------------------------------------
df = depuraTrainSet(df_train, df_target)
df = drop_null_values(df)
df = drop_neutral_values(df)
df = balanceOverSampling(df)
df = changeToNumberDependientVariable(df)
df = selectingVariablesToModel(df)
df = changeTypeToCagory(df, ['Name','Alignment'])
df = ScaleNumericVariables(df)
df = generateDummySet(df)
    
# Making X and y train/test sets
# --------------------------------------------------------------------
X = df.drop(['Name','Alignment'], axis=1).to_numpy()
y = df.Alignment.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)

# Training Logistic Regresion model
# --------------------------------------------------------------------
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))

# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Target Pipeline
# --------------------------------------------------------------------
dft = selectingVariablesToModel(df_target)
dft = changeToNumberDependientVariable(dft)
dft = changeTypeToCagory(dft, ['Name','Alignment'])
dft = ScaleNumericVariables(dft)
dft = generateDummySet(dft)

# Making predictions on target dataset
# --------------------------------------------------------------------
X_t = dft.drop(['Name','Alignment'], axis=1).to_numpy()
y_pred_t = logreg.predict(X_t)
df_target['Prediction'] = y_pred_t
df_submission = df_target[['Name','Alignment','Prediction']]
df_submission = df_submission.copy()
df_submission.Prediction = np.where(df_submission.Prediction==1,'good','bad')
print(df_submission.Alignment.value_counts())
print(df_submission.Prediction.value_counts())

# Save submission csv file to upload in kaggle
# --------------------------------------------------------------------
df_submission.to_csv(this_folder + '/outputs/submission.csv', index=False)

# plot results  of model
# --------------------------------------------------------------------
# df_submission.Alignment.value_counts().plot(kind='bar')
# df_submission.Prediction.value_counts().plot(kind='bar')

# barplot Alignment and Prediction side by side
# --------------------------------------------------------------------
x1, y1 = df_submission.Alignment.value_counts().index, df_submission.Alignment.value_counts().values
x2, y2 = df_submission.Prediction.value_counts().index, df_submission.Prediction.value_counts().values

fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
ax1.bar(x=x1, height=y1)
ax1.set_title('Alignment')
ax1.set_ylabel('Total characters')

ax2.bar(x=x2, height=y2)
ax2.set_title('Prediction')
ax2.set_ylim(0, 120)
plt.show()