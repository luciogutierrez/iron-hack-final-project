# --------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

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
df_train = pd.read_csv('datasets/marvel-powers.csv')

# --------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------
# df_train.head()
# df_train.info()
# df_train.isna().sum()
# df_train.Alignment.value_counts()
# df_train.Alignment.value_counts(normalize=True)
# df_train.columns
corr = df_train.drop('Total', axis=1).corr()
sns.heatmap(corr, annot=True, cmap='RdYlGn')

# --------------------------------------------------------------------
# Transformation
# --------------------------------------------------------------------
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
    ['Name', 'Alignment', 'Intelligence', 'Strength', 'Speed', 'Durability',
       'Power', 'Combat', 'Total']    """
    df = df.copy()
    model_cols = ['Name', 'Alignment', 'Intelligence', 'Strength', 'Speed', 'Durability',
       'Power', 'Combat']
    df = df[model_cols]
    return df

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
    df_1 = df[df.Alignment == 1]
    df_0 = df[df.Alignment == 0]
    df_resample = resample(df_0, replace=True, n_samples=df_1.shape[0])
    df_bal = pd.concat([df_resample, df_1])
    df_bal = df_bal.sample(frac=1)
    return df_bal

# Scale numerical variables
# --------------------------------------------------------------------
def ScaleNumericVariables(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['int64']).columns.tolist()
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
df = drop_null_values(df_train)
df = drop_neutral_values(df)
df = selectingVariablesToModel(df)
df = changeToNumberDependientVariable(df)
df = changeTypeToCagory(df, ['Name','Alignment'])
df = balanceOverSampling(df)

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