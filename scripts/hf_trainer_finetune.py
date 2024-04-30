#%%
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import AutoImageProcessor
from transformers import HieraForImageClassification
from transformers import HieraConfig, HieraModel
from transformers import AutoImageProcessor, HieraForPreTraining
import torch
from PIL import Image
import requests
from transformers import DefaultDataCollator
device = "cuda" if torch.cuda.is_available() else "cpu"
# %%
from datasets import Dataset
import pandas as pd
import os 

TEST_LEN = 100


df = pd.read_csv("data/folds/fold_0.csv")
df = df.rename({"class_id":"label"}, axis = 1)
df['image_path'] = df['image_path'].apply(lambda x: x if os.path.exists(x) else None)
df.dropna(subset=['image_path'], inplace=True)




# %%
#%%

#%%
# Initializing a model (with random weights) from the hiera-base-patch16-224 style configuration
# model = HieraModel(configuration)
# # Accessing the model configuration
# configuration = model.config
# %%


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("EduardoPacheco/hiera-tiny-224-mae")
model = HieraForImageClassification.from_pretrained("EduardoPacheco/hiera-tiny-224-mae", num_labels =1784 ).to(device)
#%%
# inputs = image_processor(images=image, return_tensors="pt")

# outputs = model(**inputs)
# logits = outputs.logits
# list(logits.shape)

# from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

# normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
# size = (
#     image_processor.size["shortest_edge"]
#     if "shortest_edge" in image_processor.size
#     else (image_processor.size["height"], image_processor.size["width"])
# )
# _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
def transforms(examples):
    img = Image.open(examples['image'])
    inputs = image_processor(images=img, return_tensors="pt")
    return inputs

# creating the dataset by designing custom generator 
def generator(): 
    gen_df = df.iloc[: - TEST_LEN]

    for idx, row in gen_df.iterrows(): 
        with Image.open(row['image_path']) as img:
            inputs = image_processor(images=img, return_tensors="pt")
            yield {"pixel_values": inputs['pixel_values'][0], "label": int(row['label'])}

def generate_test_data(): 
    gen_df = df.iloc[- TEST_LEN:]
    for idx, row in gen_df.iterrows(): 
        with Image.open(row['image_path']) as img:
            inputs = image_processor(images=img, return_tensors="pt")
            yield {"pixel_values": inputs['pixel_values'][0], "label": int(row['label'])}
        # img = Image.open(row['image_path'])
        # label = row['label']
        # yield {"image": img, "label": int(label)}
train_ds = Dataset.from_generator(generator)
print("TRAIN DS", len(train_ds))

eval_ds = Dataset.from_generator(generate_test_data )

print("EVAL DS", len(eval_ds))


# dataset = dataset.with_transform(transforms)
data_collator = DefaultDataCollator()
# %%
# ds = ds.with_transform(transforms)
#%%
import evaluate

accuracy = evaluate.load("accuracy")
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir = "hiera_model",
    remove_unused_columns = False,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = 5e-5,
    num_train_epochs = 3,
    per_device_train_batch_size = 64, 
    per_device_eval_batch_size = 64,
    warmup_ratio = 0.1,
    logging_steps = 10,
    load_best_model_at_end = True,
    auto_find_batch_size = True,
    metric_for_best_model = "accuracy",
    push_to_hub = True,
    push_to_hub_token=os.getenv("HF_TOKEN" , None) , 
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_ds, 
    eval_dataset=eval_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()

# %%

