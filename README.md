# ConvSDXL
A convenient library to use SDXL. Conv stands for "Convenient", not "Convolutional".

*for now, it only works for CUDA enabled systems and Colab T4 Instances*

*Note: ConvSDXL is still in development.*

## Install:
~~~
pip install git+https://github.com/vinesagar/ConvSDXL.git
~~~

## Import:
~~~
from convsdxl import ConvSDXL

sdxl = ConvSDXL()
~~~

## Get Image:
~~~
image = sdxl.get_image(prompt)
~~~

### To add negative prompt:
Provide the negative prompt as a parameter 
~~~
image = sdxl.get_image(prompt, negative_prompt="NEGATIVE PROMPT HERE")
~~~

### To save the image as you get it:
Provide a path to save it
~~~
image = sdxl.get_image(prompt, image_dir="IMAGE PATH")
~~~

### To add style to the images:
~~~
from convsdxl.enums import DesignType

image = sdxl.get_image(prompt, design_type=DesignType.StyleName)
~~~

There are few styles available for now.
- Digital Art
- Anime
- Neonpunk
- Pixel Art
- Minimalist

*Please feel free to suggest styles with their prompts and negative prompts*

### To control number of inference steps:
Simply provide 'num_inference_steps' with any integer.
~~~
image = sdxl.get_image(prompt, num_inference_steps=25)
~~~

**Any of the above mentioned parameters work with the following methods as well.**

## Inpaint Image:
Use this to make changes to an image. 
~~~
image = sdxl.inpaint_image(prompt, image, mask_image)
~~~

## Refine Image:
Use this to refine the details of an image.
~~~
image = sdxl.refine_image(prompt, image)
~~~

## Upscale Image: (Doesn't work in Colab)
Use this to enlarge the images.

*Currently, upscaling images using [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) is beyond Colab's free T4 instance's capability.*
~~~
image = sdxl.upscale_image(prompt, image)
~~~


---

## Use With Other Models

ConvSDXL can be used with other models as well by simply providing their HuggingFace handle or their loaded pipeline variable.

~~~
sdxl.set_base(model_name="HuggingFace Handle", ...)
~~~
or use an already initialized pipeline
~~~
pipe = diffusers.AnyPipeline(...)

sdxl.set_base(base_pipeline=pipe)
~~~

**You can do the same for Inpaint, Refiner and Upscaler as well.**

Have Fun!
