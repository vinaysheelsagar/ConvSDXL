import os
import torch
from datetime import datetime
from diffusers import DiffusionPipeline, AutoencoderKL
from .enums import *

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae, torch_dtype=torch.float16, variant="fp16",
    use_safetensors=True
)

_ = pipe.to("cuda")

def __clean_prompt(prompt: str):
  cleaned_prompt = prompt.strip()
  return cleaned_prompt


def __save_image(image, image_dir):
  image_path = os.path.join(image_dir, f"{datetime.now()}.png")
  image.save(image_path, "PNG")


def add_design_to_prompt(prompt: str, negative_prompt: str, design_type: DesignType):
    match design_type:
        case DesignType.DigitalArt:
            designed_prompt = f"""concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed"""
            designed_negative_prompt =  f"""{negative_prompt} photo, photorealistic, realism, ugly"""

        case DesignType.Anime:
            designed_prompt = f"""anime artwork {prompt} . anime style, key visual, vibrant, studio anime, highly detailed"""
            designed_negative_prompt =  f"""{negative_prompt} photo, deformed, black and white, realism, disfigured, low contrast"""

        case DesignType.Neonpunk:
            designed_prompt = f"""neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional"""
            designed_negative_prompt =  f"""{negative_prompt} painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"""

        case DesignType.PixelArt:
            designed_prompt = f"""pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics"""
            designed_negative_prompt =  f"""{negative_prompt} sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic"""

        case DesignType.Minimalist:
            designed_prompt = f"""minimalist style {prompt} . simple, clean, uncluttered, modern, elegant"""
            designed_negative_prompt =  f"""{negative_prompt} ornate, complicated, highly detailed, cluttered, disordered, messy, noisy"""

        case _:
            raise ValueError("No such DesignType")

    return designed_prompt, designed_negative_prompt


def get_image(
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        design_type: DesignType = None,
        image_dir: str = None,
):
    os.makedirs(image_dir, exist_ok=True)

    _prompt = __clean_prompt(prompt)
    _negative_prompt = __clean_prompt(negative_prompt)

    if design_type is not None:
        designed_prompts = add_design_to_prompt(_prompt, _negative_prompt, design_type)

        _prompt = designed_prompts[0]
        _negative_prompt = designed_prompts[1]

    image = pipe(
        prompt=_prompt,
        negative_prompt=_negative_prompt,
        num_inference_steps=num_inference_steps,
    ).images[0]

    if image_dir is not None:
        __save_image(image, image_dir)

    return image
