import os
import gc
import torch
from .enums import *
from PIL import Image
from datetime import datetime
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    StableDiffusionXLImg2ImgPipeline)


_base: StableDiffusionXLPipeline | None = None
_refiner: StableDiffusionXLImg2ImgPipeline | None = None


def _clear_garbage():
    gc.collect()


def set_base(
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        vae: str = "madebyollin/sdxl-vae-fp16-fix",
        base=None,
):
    global _base

    if base is not None:
        _base = base
        return

    _clear_garbage()

    vae = AutoencoderKL.from_pretrained(vae, torch_dtype=torch.float16)

    _base = DiffusionPipeline.from_pretrained(
        model,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    _base.to("cuda")


def remove_base():
    global _base
    _base = None
    torch.cuda.empty_cache()
    _clear_garbage()


def set_refiner(
        model: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
        vae: str = "madebyollin/sdxl-vae-fp16-fix",
        refiner=None,
):
    global _refiner

    if refiner is not None:
        _refiner = refiner
        return

    _clear_garbage()

    vae = AutoencoderKL.from_pretrained(vae, torch_dtype=torch.float16)

    _refiner = DiffusionPipeline.from_pretrained(
        model,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    _refiner.to("cuda")
    _refiner()


def remove_refiner():
    global _refiner
    _refiner = None
    torch.cuda.empty_cache()
    _clear_garbage()


def _clean_prompt(prompt: str):
    cleaned_prompt = prompt.strip()
    return cleaned_prompt


def _save_image(image, image_dir):
    image_path = os.path.join(image_dir, f"{datetime.now()}.png")
    image.save(image_path, "PNG")


def add_design_to_prompt(design_type: DesignType, prompt: str | None = None, negative_prompt: str | None = None):

    if prompt is None:
        prompt = ""

    if negative_prompt is None:
        negative_prompt = ""

    # TODO: Better management of strings
    match design_type:
        case DesignType.DigitalArt:
            designed_prompt = f"""concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed"""
            designed_negative_prompt = f"""{negative_prompt} photo, photorealistic, realism, ugly"""

        case DesignType.Anime:
            designed_prompt = f"""anime artwork {prompt} . anime style, key visual, vibrant, studio anime, highly detailed"""
            designed_negative_prompt = f"""{negative_prompt} photo, deformed, black and white, realism, disfigured, low contrast"""

        case DesignType.Neonpunk:
            designed_prompt = f"""neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional"""
            designed_negative_prompt = f"""{negative_prompt} painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"""

        case DesignType.PixelArt:
            designed_prompt = f"""pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics"""
            designed_negative_prompt = f"""{negative_prompt} sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic"""

        case DesignType.Minimalist:
            designed_prompt = f"""minimalist style {prompt} . simple, clean, uncluttered, modern, elegant"""
            designed_negative_prompt = f"""{negative_prompt} ornate, complicated, highly detailed, cluttered, disordered, messy, noisy"""

        case _:
            raise ValueError("No such DesignType")

    return designed_prompt, designed_negative_prompt


def get_image(
        prompt: str,
        negative_prompt: str = None,
        num_inference_steps: int = 25,
        design_type: DesignType = None,
        image_dir: str = None,
        **model_kwargs
):
    global _base

    os.makedirs(image_dir, exist_ok=True)

    if _base is None:
        set_base()

    prompt = _clean_prompt(prompt)
    if negative_prompt is not None:
        negative_prompt = _clean_prompt(negative_prompt)
        model_kwargs["negative_prompt"] = negative_prompt

    if design_type is not None:
        designed_prompts = add_design_to_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            design_type=design_type,
        )

        prompt = designed_prompts[0]
        negative_prompt = designed_prompts[1]
        model_kwargs["negative_prompt"] = negative_prompt

    image = _base(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        **model_kwargs
    ).images[0]

    if image_dir is not None:
        _save_image(image, image_dir)

    return image


def refine_image(
        image: Image,
        prompt: str,
        negative_prompt: str = None,
        num_inference_steps: int = 25,
        design_type: DesignType = None,
        image_dir: str = None,
        **model_kwargs,
):
    global _refiner

    os.makedirs(image_dir, exist_ok=True)

    if _refiner is None:
        set_refiner()

    prompt = _clean_prompt(prompt)
    if negative_prompt is not None:
        negative_prompt = _clean_prompt(negative_prompt)
        model_kwargs["negative_prompt"] = negative_prompt

    if design_type is not None:
        designed_prompts = add_design_to_prompt(
            design_type=design_type,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

        prompt = designed_prompts[0]
        negative_prompt = designed_prompts[1]
        model_kwargs["negative_prompt"] = negative_prompt

    image = _refiner(
        image=image,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        **model_kwargs,
    ).images[0]

    if image_dir is not None:
        _save_image(image, image_dir)

    return image
