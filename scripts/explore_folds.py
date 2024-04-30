#%% 
import pandas as pd    
import matplotlib.pyplot as plt 
import random 

fold_1 = pd.read_csv("data/folds/fold_1.csv") 
# %%
plt.imshow(plt.imread(random.choice(fold_1.image_path)))
# %%
