import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

# 1. Model Setup
model_id = "stabilityai/sd-turbo" # Fast generation ke liye ye best hai
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe = pipe.to(device)

# 2. Prediction Function
def generate_design(prompt, negative_prompt):
    # Free CPU par steps kam rakhein taaki jaldi generate ho
    image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=4, # SD-Turbo ke liye 4 steps kaafi hain
        guidance_scale=0.0
    ).images[0]
    return image

# 3. Gradio Interface Setup
interface = gr.Interface(
    fn=generate_design,
    inputs=[
        gr.Textbox(label="Describe your design", placeholder="A red silk evening dress..."),
        gr.Textbox(label="What to avoid", value="ugly, blurry, low quality")
    ],
    outputs=gr.Image(label="Generated Fashion Design"),
    title="👗 AI Fashion Design Generator",
    description="Apna design describe karein aur AI use generate karega."
)

# 4. Launch
if __name__ == "__main__":
    interface.launch()
