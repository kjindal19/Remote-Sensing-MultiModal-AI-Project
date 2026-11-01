import torch
import datasets
from tqdm import tqdm 

from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 



model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base-ft",
    trust_remote_code=True,
    revision='refs/pr/6'
).to(device) 

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", 
    trust_remote_code=True, revision='refs/pr/6')

for param in model.vision_tower.parameters():
  param.is_trainable = False

import datasets
import json
import pandas as pd
from PIL import Image

# Load data from JSON files
with open('data/all_questions_variated.json', 'r') as f:
    questions_data = json.load(f)
with open('data/all_answers.json', 'r') as f:
    answers_data = json.load(f)

# Create Pandas DataFrames
questions_df = pd.DataFrame(questions_data['questions'])
answers_df = pd.DataFrame(answers_data['answers'])

# Join DataFrames based on question_id
df = questions_df.join(answers_df, on='id', lsuffix='_question')
df = df[['id','type','question','answer','img_id']]

training_df = df.iloc[:500]
validation_df = df.iloc[500:600]


# Create a Dataset
train_dataset = datasets.Dataset.from_pandas(training_df)
validation_dataset = datasets.Dataset.from_pandas(validation_df)

# Add image column
def load_image(obj):
    obj['image'] = Image.open(f'data/Images/{obj["img_id"]}.tif')
    return obj

train_dataset = train_dataset.map(load_image)
validation_dataset = validation_dataset.map(load_image)

# Dataset now contains image, question, and answer columns
print(train_dataset)
print(validation_dataset)


#pending changes



import torch 
from torch.utils.data import Dataset 



class RSVQADataset(Dataset):
    def __init__(self,data):
        self.data = data
        print(data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        example = self.data[idx]
        question = "<RSVQA>" + example['question'] 
        first_answer = example['answer']
        image = example['image'].convert("RGB")
        return question, first_answer, image
    
import os 
from torch.utils.data import DataLoader
from tqdm import tqdm 
from transformers import get_scheduler

def collate_fn(batch): 
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers 

train_dataset = RSVQADataset(train_dataset)
val_dataset = RSVQADataset(validation_dataset) 
batch_size = 9
num_workers = 0



train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          collate_fn=collate_fn, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          collate_fn=collate_fn, num_workers=num_workers, shuffle=True)




epochs = 15
optimizer = AdamW(model.parameters(), lr=1e-6)
num_training_steps = epochs * len(train_loader)

lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, 
                              num_warmup_steps=0, num_training_steps=num_training_steps,)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def calculate_iou(predictions, references):
    intersection = np.logical_and(predictions, references)
    union = np.logical_or(predictions, references)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score



def calculate_metrics(predictions, references):
    accuracy = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average='weighted')
    precision = precision_score(references, predictions, average='weighted')
    recall = recall_score(references, predictions, average='weighted')
    iou_score = calculate_iou(predictions, references)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'iou': iou_score
    }

def evaluate_model(model, val_loader, processor, device):
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs, answers = batch
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
            
            outputs = model(input_ids=input_ids, pixel_values=pixel_values)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            predictions.extend(preds.cpu().numpy())
            references.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    references = np.array(references).flatten()
    
    metrics = calculate_metrics(predictions, references)
    return metrics

# Evaluate the model
metrics = evaluate_model(model, val_loader, processor, device)
print(metrics)


for epoch in range(epochs): 
    model.train() 
    train_loss = 0
    i = -1
    for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
        i += 1
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"] 
        labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    print(f"Average Training Loss: {avg_train_loss}")

    model.eval()
    val_loss = 0
    j= 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
            inputs, answers = batch
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            j += 1
            if j % 400 == 0:
                print(val_loss / len(val_loader))


    metrics = evaluate_model(model, val_loader, processor, device)
    print(metrics)
    model.save_pretrained(f"./RSVQA-florence2-{epoch+1}")
    processor.save_pretrained(f"./RSVQA-florence2-{epoch+1}")

'''def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

def process_image(image, text_input=None):
    image = Image.fromarray(image)  # Convert NumPy array to PIL Image
    task_prompt = '<DocVQA>'
    results = run_example(task_prompt, image, text_input)[task_prompt].replace("<pad>", "")
    return results


demoimg = Image.open('/data/Images/10.tif')
question = "Is it a rural or an urban area"
prompt = '<RSVQA>'+question

print(run_example(prompt,demoimg)[prompt].replace("<pad>", ""))

'''