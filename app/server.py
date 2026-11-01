from flask import Flask, request, jsonify
from PIL import Image
import io
import time

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

app = Flask(__name__)




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



# Dummy function to simulate answer generation
def answer_question(image: Image.Image, question: str) -> str:
    image = Image.fromarray(image)  # Convert NumPy array to PIL Image
    task_prompt = '<RSVQA>'
    results = run_example(task_prompt, image, question)[task_prompt].replace("<pad>", "")
    return results

@app.route('/vqa', methods=['POST'])
def vqa():
    if 'image' not in request.files or 'question' not in request.form:
        return jsonify({'error': 'Missing image or question'}), 400

    image_file = request.files['image']
    question = request.form['question']

    try:
        image = Image.open(image_file.stream)
        answer = answer_question(image, question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
