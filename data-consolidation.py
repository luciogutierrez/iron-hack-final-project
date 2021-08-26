# imports
import pandas as pd

# Data consolidation
# --------------------------------------------------------------------

# Extraction
# --------------------------------------------------------------------
data1 = pd.read_csv('./datasets/marvel_character_info.csv')
data2 = pd.read_csv('./datasets/marvel_powers_matrix.csv')

# Data concatenation
# --------------------------------------------------------------------
data = pd.merge(data1, data2, on=['Name','Alignment'])

# Generate train set with Marvel characters
# --------------------------------------------------------------------
data3 = data[data.Publisher=='Marvel Comics']
data3.to_csv('./datasets/marvel_data.csv', index=False)

# Generate target set with DC characters
# --------------------------------------------------------------------
data4 = data[data.Publisher=='DC Comics']
data4.to_csv('./datasets/dc_data.csv', index=False)
