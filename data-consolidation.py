# imports
import os
import pandas as pd

# Data consolidation
# --------------------------------------------------------------------
this_folder = os.path.dirname(os.path.abspath(__file__))

# Extraction
# --------------------------------------------------------------------
data1 = pd.read_csv(this_folder + '/datasets/marvel_character_info.csv')
data2 = pd.read_csv(this_folder + '/datasets/marvel_powers_matrix.csv')

# Data concatenation
# --------------------------------------------------------------------
data = pd.merge(data1, data2, on=['Name','Alignment'])

# Generate train set with Marvel characters
# --------------------------------------------------------------------
data3 = data[data.Publisher=='Marvel Comics']
data3.to_csv(this_folder + '/datasets/marvel_data.csv', index=False)

# Generate target set with DC characters
# --------------------------------------------------------------------
data4 = data[data.Publisher=='DC Comics']
data4.to_csv(this_folder + '/datasets/dc_data.csv', index=False)
