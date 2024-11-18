import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline


@st.cache_resource
def load_model(model_path="Lykon/dreamShaper"):
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
    pipe.to("mps")
    return pipe

def generate_image(prompt, pipe):
    return pipe(prompt).images[0]

st.title("GenVision")
prompt = st.text_input("Enter your type image you want to generate:")
if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating Image..."):
            pipe = load_model()
            image = generate_image(prompt, pipe)
    else:
        st.warning("Please enter a prompt!!!")
