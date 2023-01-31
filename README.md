# Open Journey Cog model

A fork of [cog-stable-diffusion](https://github.com/replicate/cog-stable-diffusion) to deploy a version of Prompt Hero's OpenJourney with support for negative prompts and NSFW filter.

First, download the pre-trained weights:

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i prompt="monkey scuba diving"
