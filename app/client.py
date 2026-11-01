import gradio as gr
import requests

API_URL = "http://127.0.0.1:5000/vqa"  # Change if hosted elsewhere

def query_server(image, question):
    if not image or not question:
        return "Please provide both an image and a question."

    # Convert image to byte stream
    _, img_encoded = image.save("temp.png", format="PNG"), None
    with open("temp.png", "rb") as img_file:
        files = {'image': img_file}
        data = {'question': question}
        try:
            response = requests.post(API_URL, files=files, data=data)
            return response.json().get('answer', "No answer received.")
        except Exception as e:
            return f"Error: {e}"

iface = gr.Interface(
    fn=query_server,
    inputs=[
        gr.Image(type="pil", label="Upload Satellite Image"),
        gr.Textbox(lines=2, label="Enter your question")
    ],
    outputs="text",
    title="Remote Sensing VQA",
    description="Ask a question about a satellite image!"
)

if __name__ == "__main__":
    iface.launch()
