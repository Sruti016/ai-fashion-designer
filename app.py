import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import os

# --- 1. CONFIGURATION ---
MY_HF_TOKEN = st.secrets["HF_TOKEN"] # Ensure this is active!

st.set_page_config(page_title="AI Fashion Designer", page_icon="👗", layout="wide")
st.title("👗 AI Fashion Design Generator")
st.markdown("Describe a clothing item, and our AI will generate a unique design for you.")

# --- 2. MODEL SETUP ---
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"

@st.cache_resource
def load_pipeline(token):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_auth_token=token
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing() # Crucial for Colab
    return pipe

# --- 3. UI LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_area("Describe your design:",
                         "A high fashion elegant red silk evening dress, 4k resolution, cinematic lighting")
    negative_prompt = st.text_input("What to avoid:", "ugly, deformed, low quality, blurry")
    generate_btn = st.button("Generate Design 🎨", type="primary")

# --- 4. GENERATION LOGIC ---
if generate_btn:
    with st.spinner("✨ Designing your outfit... (This takes about 20-30 seconds)"):
        try:
            pipe = load_pipeline(MY_HF_TOKEN)

            # Generate Image
            image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=50).images[0]

            # Display Results
            st.image(image, caption="Your AI Generated Design", use_column_width=True)

            with col2:
                st.subheader("🛍️ Shop Similar Styles")
                search_query = prompt.replace(" ", "+")
                st.markdown(f"🔍 [Search on Amazon](https://www.amazon.com/s?k={search_query})")
                st.markdown(f"🔍 [Search on eBay](https://www.ebay.com/sch/i.html?_nkw={search_query})")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            if "401" in str(e):
                st.warning("Hint: This looks like a Hugging Face Token error. Check your token permissions.")
