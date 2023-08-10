import pandas as pd
import numpy as np


A = pd.DataFrame(data={"a": [1, 2, np.NaN], "b": [4, np.NaN, 6]})

B = pd.DataFrame(data={"a": [1, 8, 7], "c": [4, 9, 6]})

# Set the index of B to [11, 12, 13]
B.index = [11, 12, 13]


# Now fill all values in A with the values in B, where A is NaN
A.fillna(B, inplace=True)
