#%%
from fastai.vision.all import *
import pandas as pd 

#%
df = pd.read_csv("data/folds/fold_0.csv")
df = df.rename({"class_id":"label"}, axis = 1)
# verify if the image_path exists 
df['image_path'] = df['image_path'].apply(lambda x: x if os.path.exists(x) else None)
# drop if the image_path is None
print(len(df))
df = df.dropna(subset=['image_path'])
print(len(df))

datablock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),  # Define the type of input and output blocks
    get_x=ColReader('image_path'),       # Function to get the image files
    get_y=ColReader('label'),            # Function to get the labels
    splitter=RandomSplitter(valid_pct=0.001),  # Split data into training and validation sets
    item_tfms=Resize(224)                # Resize images to 224x224 pixels
)

dls = datablock.dataloaders(df)

# %%
learn = vision_learner(dls, "resnet18", metrics=accuracy)
learn.fine_tune(6)


# %%
