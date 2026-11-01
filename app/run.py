import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


model = AutoModelForCausalLM.from_pretrained(
    "./model",
    trust_remote_code=True,
    ).to(device) 
for param in model.vision_tower.parameters():
    param.is_trainable = False
processor = AutoProcessor.from_pretrained("./model", 
    trust_remote_code=True)

def run_example(task_prompt, image, text_input=None):
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

demoimg = Image.open('/kaggle/input/rs-dataset/Images_LR/Images_LR/10.tif')
question = "Is it a rural or an urban area"
prompt = '<RSVQA>'

print(run_example(prompt,demoimg,question)[prompt].replace("<pad>", ""))

