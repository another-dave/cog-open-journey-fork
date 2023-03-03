import os
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (StableDiffusionSafetyChecker)
from PIL import ImageOps
import gc
import pprint


MODEL_ID = "prompthero/openjourney"
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"

def report_gpu():
    print("******************************************************")
    print("********      Start report_gpu()        **************")
    print("******************************************************")
    pp = pprint.PrettyPrinter(depth=4)
    print("******************************************************")
    print("******** Stats before GC and EmptyCache **************")
    print("******************************************************")
    pp.pprint(torch.cuda.list_gpu_processes())
    pp.pprint(torch.cuda.memory_stats())
    pp.pprint(torch.cuda.memory_summary())
    gc.collect()
    torch.backends.cuda.cufft_plan_cache.clear()
    torch.cuda.empty_cache()
    print("******************************************************")
    print("******** Stats after GC and EmptyCache  **************")
    print("******************************************************")
    pp.pprint(torch.cuda.list_gpu_processes())
    pp.pprint(torch.cuda.memory_stats())
    pp.pprint(torch.cuda.memory_summary())
    print("******************************************************")
    print("********      End report_gpu()          **************")
    print("******************************************************")

class Predictor(BasePredictor):
    def setup(self):
        print("******************************************************")
        print("********             setup()            **************")
        print("******************************************************")
        report_gpu()
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt (Input an array of prompts with | separator)",
            default="a photo of an astronaut riding a horse on mars.",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output. Only need one negative prompt regardlses of prompt count (will be repeated for each)",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        contrast_cutoff: int = Input(
            description="How much auto-contrast to apply to output images (using PIL)",
            ge=0,
            le=100,
            default=2,
        ),
    ) -> List[Path]:

        print("******************************************************")
        print("********           predict()            **************")
        print("******************************************************")
        report_gpu()

        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)

        prompts = prompt.split("|") if prompt is not None else None

        prompts_count = len(prompts) if prompt is not None else 0 

        print(f"Raw Input Prompt: {prompt}")
        print(f"Count: {prompts_count}")
        print(f"Prompts: {prompts}")

        print(f"Negative Prompt: {negative_prompt}")

        prompts = [prompt] * num_outputs if prompts_count == 1 else prompts if prompt is not None else None
        negative_prompts = ([negative_prompt] * (num_outputs if prompts_count == 1 else prompts_count) ) if negative_prompt is not None else None

        output = self.pipe(
            prompt=prompts,
            negative_prompt=negative_prompts,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        print(f"Output image count: {len(output.images)}")

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                continue

            output_image = sample if contrast_cutoff == 0 else ImageOps.autocontrast(sample, cutoff = contrast_cutoff, ignore = 0)

            output_path = f"/tmp/out-{i}.png"
            output_image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
