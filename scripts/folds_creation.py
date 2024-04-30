#%%
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import model_selection
import pandas as pd
import random 

train_metadata = pd.read_csv("data/SnakeCLEF2023-TrainMetadata-iNat.csv")
rare_classes = pd.read_csv("data/SnakeCLEF2023-TrainMetadata-HM.csv")
val_metadata = pd.read_csv("/teamspace/studios/this_studio/data/SnakeCLEF2023-ValMetadata.csv")

base_path = "data/SnakeCLEF2023-small_size/"
rare_images_path = "data/"


train_metadata["image_path"] = base_path + train_metadata["image_path"]
val_metadata['image_path'] = base_path + val_metadata['image_path']
rare_classes['image_path'] = rare_images_path + rare_classes['image_path']


total_data = pd.concat([train_metadata, val_metadata, rare_classes]).reset_index(drop = True)

#%%
total_data["kfold"] = -1
y = total_data.binomial_name

skf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

idx = 0 
for fold_no, (train_idx, test_idx) in enumerate(skf.split(total_data, total_data.binomial_name)): 
    total_data.loc[test_idx, "kfold"] = fold_no
    total_data.loc[test_idx].to_csv(f"data/folds/fold_{fold_no}.csv", index = False)
    print(f"Fold {fold_no} created")

for x in range(10): 
    random_image = random.choice(total_data.image_path) 
    img = plt.imread(random_image)
    plt.imshow(img)
