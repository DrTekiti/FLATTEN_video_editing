import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import torch
import torchvision
from models.pipeline_flatten import FlattenPipeline
from models.util import save_videos_grid, read_video, sample_trajectories
from models.unet import UNet3DConditionModel


class Flatten:

    @classmethod
    def INPUT_TYPES(cls):
               
        return {"required": {
                    "model": ("MODEL",),
                    "video": ("STRING", {"default": "X://insert/path/here.mp4" }),
                    "output_path": ("STRING", {"default": "X://insert/path/here"}),
                    "frame_rate": ("INT", {"default": 24, "min": 1, "max": 24, "step": 1}),
                    "video_length": ("INT", {"default": 15, "min": 1, "max": 60, "step": 1}),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "unsample_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.1}),
                    "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step":32}), 
                    "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step":32}),
                    "inject_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "old_qk": ("INT", {"default": 0, "min": 0, "max": 1}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "vae": ("VAE", ),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    #RETURN_NAMES = ("IMAGE",)
    FUNCTION = "sFlatten"
    CATEGORY = "FLATTEN"

    def sFlatten(self):
        tokenizer = CLIPTokenizer.from_pretrained(self.model, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.model, subfolder="text_encoder").to(dtype=torch.float16)
        vae = AutoencoderKL.from_pretrained(self.model, subfolder="vae").to(dtype=torch.float16)
        unet = UNet3DConditionModel.from_pretrained_2d(self.model, subfolder="unet").to(dtype=torch.float16)
        scheduler=DDIMScheduler.from_pretrained(self.model, subfolder="scheduler")
        inverse=DDIMInverseScheduler.from_pretrained(self.model, subfolder="scheduler")

        pipe = FlattenPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=scheduler, inverse_scheduler=inverse)
        pipe.enable_vae_slicing()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.to(device)

        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)

        # read the source video
        video = read_video(video_path=self.video, video_length=self.video_length,
                        width=self.width, height=self.height, frame_rate=self.frame_rate)
        original_pixels = rearrange(video, "(b f) c h w -> b c f h w", b=1)
        save_videos_grid(original_pixels, os.path.join(self.output_path, "source_video.mp4"), rescale=True)

        t2i_transform = torchvision.transforms.ToPILImage()
        real_frames = []
        for i, frame in enumerate(video):
            real_frames.append(t2i_transform(((frame+1)/2*255).to(torch.uint8)))

        # compute optical flows and sample trajectories
        trajectories = sample_trajectories(os.path.join(self.output_path, "source_video.mp4"), device)
        torch.cuda.empty_cache()

        for k in trajectories.keys():
            trajectories[k] = trajectories[k].to(device)
        sample = pipe(self.positive, video_length=self.video_length, frames=real_frames,
                    num_inference_steps=self.steps, generator=generator, guidance_scale=self.cfg,
                    negative_prompt=self.negative, width=self.width, height=self.height,
                    trajs=trajectories, output_dir="tmp/", inject_step=self.inject_step, old_qk=self.old_qk).videos
        temp_video_name = self.positive+"_"+self.negative+"_"+str(self.cfg)
        save_videos_grid(sample, f"{self.output_path}/{temp_video_name}.mp4", fps=self.fps)    
         

        return (({sample}, ))





NODE_CLASS_MAPPINGS = { 
    "Flatten": Flatten,
    }
    