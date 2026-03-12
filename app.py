import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import urllib.parse

# 1. Model Setup (Fast)
model_id = "stabilityai/sd-turbo"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe = pipe.to(device)

def generate_fashion(prompt):
    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
    query = urllib.parse.quote(prompt)
    amazon_link = f"https://www.amazon.in/s?k={query}"
    flipkart_link = f"https://www.flipkart.com/search?q={query}"
    
    shopping_html = f"""
    <div style="display: flex; gap: 15px; justify-content: center; margin-top: 20px;">
        <a href="{amazon_link}" target="_blank" class="shop-btn amazon">🛒 View on Amazon</a>
        <a href="{flipkart_link}" target="_blank" class="shop-btn flipkart">🛍️ View on Flipkart</a>
    </div>
    """
    return image, shopping_html

# 2. Custom CSS for Styling
custom_css = """
/* Poore page ka background */
.gradio-container {
    background: linear-gradient(135deg, #fff5f7 0%, #ffffff 100%);
    font-family: 'Poppins', sans-serif;
}

/* Title styling */
#title { 
    text-align: center; 
    color: #C2185B; 
    font-size: 2.5em; 
    font-weight: 800;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 0px;
}

/* Input box aur Image ko thoda 'Card' wala look dene ke liye */
.gr-box {
    border-radius: 15px !important;
    border: 1px solid #fce4ec !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
}

/* Shopping buttons styling */
.shop-btn {
    padding: 12px 25px;
    border-radius: 50px;
    text-decoration: none;
    font-weight: bold;
    transition: all 0.3s ease;
    color: white !important;
    display: inline-block;
}

.shop-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}

.amazon { background: linear-gradient(90deg, #FF9900, #FFB300); }
.flipkart { background: linear-gradient(90deg, #2874F0, #047BD5); }
"""

# 3. Gradio Interface with Theme

with gr.Blocks(theme=gr.themes.Soft(primary_hue="pink", font=[gr.themes.GoogleFont("Poppins")]), css=custom_css) as demo:
    gr.HTML("<h1 id='title'>✨ AI Fashion Designer Studio ✨</h1>")
    gr.Markdown("<p style='text-align: center;'>Visualize your dream outfit in seconds. Enter a description and let the AI create magic!</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Outfit Description", 
                placeholder="e.g. A royal velvet lehenga with gold embroidery...",
                lines=4
            )
            btn = gr.Button("🎨 Generate Masterpiece", variant="primary")
            gr.Examples(
                examples=["A floral summer midi dress", "A black tuxedo with silk lapels", "An elegant wine-colored evening gown"],
                inputs=input_text
            )
        
        with gr.Column(scale=1):
            output_img = gr.Image(label="AI Visualization", elem_id="output-img")
            output_html = gr.HTML()

    btn.click(fn=generate_fashion, inputs=input_text, outputs=[output_img, output_html])
    gr.HTML("<hr><p style='text-align: center;'>Created for Fashion Enthusiasts | 2026</p>")

demo.launch()
