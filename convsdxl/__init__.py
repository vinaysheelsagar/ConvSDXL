import os
import gc
import torch
from .enums import *
from PIL import Image
from datetime import datetime
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers import StableDiffusionUpscalePipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    StableDiffusionXLImg2ImgPipeline)


def _clean_prompt(prompt: str):
    cleaned_prompt = prompt.strip()
    return cleaned_prompt


def _clear_garbage():
    gc.collect()


def _save_image(image, image_dir):
    image_path = os.path.join(image_dir, f"{datetime.now()}.png")
    image.save(image_path, "PNG")


def add_design_to_prompt(
        design_type: DesignType,
        prompt: str | None = None,
        negative_prompt: str | None = None,
):
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


def preprocess_prompts(
        design_type: DesignType | None,
        prompt: str | None,
        negative_prompt: str | None,
        **model_kwargs
):
    if prompt is not None:
        prompt = _clean_prompt(prompt)
        model_kwargs["prompt"] = prompt

    if negative_prompt is not None:
        negative_prompt = _clean_prompt(negative_prompt)
        model_kwargs["negative_prompt"] = negative_prompt

    if design_type is not None:
        designed_prompts = add_design_to_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            design_type=design_type,
        )

        model_kwargs["prompt"] = designed_prompts[0]
        model_kwargs["negative_prompt"] = designed_prompts[1]

    return model_kwargs


class ConvSDXL:
    _base: StableDiffusionXLPipeline | None = None
    _refiner: StableDiffusionXLImg2ImgPipeline | None = None
    _inpaint: StableDiffusionXLPipeline | None = None
    _upscaler: StableDiffusionUpscalePipeline | None = None

    def set_base(
            self,
            model: str = "stabilityai/stable-diffusion-xl-base-1.0",
            # vae: str = "madebyollin/sdxl-vae-fp16-fix",
            base=None,
    ):

        if base is not None:
            self._base = base
            return

        _clear_garbage()

        # vae = AutoencoderKL.from_pretrained(vae, torch_dtype=torch.float16)

        self._base = DiffusionPipeline.from_pretrained(
            model,
            # vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self._base.to("cuda")

    def remove_base(self):
        self._base = None
        torch.cuda.empty_cache()
        _clear_garbage()

    def set_inpaint(
            self,
            model: str = "stabilityai/stable-diffusion-2-inpainting",
            # vae: str = "madebyollin/sdxl-vae-fp16-fix",
            inpaint=None,
    ):

        if inpaint is not None:
            self._inpaint = inpaint
            return

        _clear_garbage()

        # vae = AutoencoderKL.from_pretrained(vae, torch_dtype=torch.float16)

        self._inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            model,
            # vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self._inpaint.to("cuda")

    def remove_inpaint(self):
        self._inpaint = None
        torch.cuda.empty_cache()
        _clear_garbage()

    def set_refiner(
            self,
            model: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
            # vae: str = "madebyollin/sdxl-vae-fp16-fix",
            refiner=None,
    ):

        if refiner is not None:
            self._refiner = refiner
            return

        _clear_garbage()

        # vae = AutoencoderKL.from_pretrained(vae, torch_dtype=torch.float16)

        self._refiner = DiffusionPipeline.from_pretrained(
            model,
            # vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self._refiner.to("cuda")

    def remove_refiner(self):
        self._refiner = None
        torch.cuda.empty_cache()
        _clear_garbage()

    def set_upscaler(
            self,
            model: str = "stabilityai/stable-diffusion-x4-upscaler",
            # vae: str = "madebyollin/sdxl-vae-fp16-fix",
            upscaler=None,
    ):

        if upscaler is not None:
            self._upscaler = upscaler
            return

        _clear_garbage()

        # vae = AutoencoderKL.from_pretrained(vae, torch_dtype=torch.float16)

        self._refiner = DiffusionPipeline.from_pretrained(
            model,
            # vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self._refiner.to("cuda")

    def remove_upscaler(self):
        self._refiner = None
        torch.cuda.empty_cache()
        _clear_garbage()

    def get_image(
            self,
            prompt: str,
            negative_prompt: str = None,
            design_type: DesignType = None,
            image_dir: str | os.PathLike = None,
            **model_kwargs
    ):

        os.makedirs(image_dir, exist_ok=True)

        if self._base is None:
            self.set_base()

        model_kwargs = preprocess_prompts(
            design_type=design_type,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

        image = self._base(
            **model_kwargs
        ).images[0]

        if image_dir is not None:
            _save_image(image, image_dir)

        return image

    def inpaint_image(
            self,
            prompt: str,
            image: Image,
            mask_image: Image,
            negative_prompt: str = None,
            design_type: DesignType = None,
            image_dir: str | os.PathLike = None,
            **model_kwargs,
    ):

        os.makedirs(image_dir, exist_ok=True)

        if self._inpaint is None:
            self.set_inpaint()

        model_kwargs = preprocess_prompts(
            design_type=design_type,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

        image = self._inpaint(
            image=image,
            mask_image=mask_image,
            **model_kwargs
        ).images[0]

        if image_dir is not None:
            _save_image(image, image_dir)

        return image

    def refine_image(
            self,
            image: Image,
            prompt: str,
            negative_prompt: str = None,
            design_type: DesignType = None,
            image_dir: str | os.PathLike = None,
            **model_kwargs,
    ):

        os.makedirs(image_dir, exist_ok=True)

        if self._refiner is None:
            self.set_refiner()

        model_kwargs = preprocess_prompts(
            design_type=design_type,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

        image = self._refiner(
            image=image,
            **model_kwargs,
        ).images[0]

        if image_dir is not None:
            _save_image(image, image_dir)

        return image

    def upscale_image(
            self,
            image: Image,
            prompt: str,
            negative_prompt: str = None,
            design_type: DesignType = None,
            image_dir: str | os.PathLike = None,
            **model_kwargs,
    ):

        os.makedirs(image_dir, exist_ok=True)

        if self._refiner is None:
            self.set_refiner()

        model_kwargs = preprocess_prompts(
            design_type=design_type,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

        image = self._refiner(
            image=image,
            **model_kwargs,
        ).images[0]

        if image_dir is not None:
            _save_image(image, image_dir)

        return image
