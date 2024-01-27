# ConvSDXL
A convenient library to use SDXL

*for now, it only works for Colab T4 Instances/CUDA enabled systems*

## Install:
~~~
pip install git@https://github.com/vinesagar/ConvSDXL.git
~~~

## Imports:
~~~
import convsdxl
~~~

## Get Image:
~~~
image = convsdxl.get_image(prompt)
~~~

## To add negative prompt:
Simply provide the negative prompt after the main prompt
~~~
image = convsdxl.get_image(prompt, negative_prompt)
~~~
Or provide the negative prompt to *negative_prompt* parameter 
~~~
image = convsdxl.get_image(prompt, negative_prompt="NEGATIVE PROMPT HERE")
~~~

## Simply provide a path to save it as well:
~~~
image = convsdxl.get_image(prompt, negative_prompt, image_dir="IMAGE PATH")
~~~

## To add style to the images:
~~~
from convsdxl.enums import DesignType

image = convsdxl.get_image(prompt, negative_prompt, design_type=DesignType.StyleName)
~~~

There are few styles available for now.
- Digital Art
- Anime
- Neonpunk
- Pixel Art
- Minimalist

*Please feel free to suggest styles with their prompts and negative prompts*

## To control number of inference steps:
Simply provide 'num_inference_steps' with any integer.
~~~
image = convsdxl.get_image(prompt, num_inference_steps=25)
~~~

Have Fun!
