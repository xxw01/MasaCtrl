"""
Util functions based on Diffuser framework.
"""


import os
import torch
import cv2
import numpy as np

import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from torchvision.io import read_image

from diffusers import StableDiffusionPipeline

from pytorch_lightning import seed_everything


class MasaCtrlPipeline(StableDiffusionPipeline):

    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # posterior variance
        posterior_variance = (1.0 - (alpha_prod_t / alpha_prod_t_prev)) * (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t)
        posterior_variance = max(posterior_variance, 1e-20)  # 数值稳定
        posterior_std = posterior_variance**0.5
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev - eta**2 * posterior_variance)**0.5 * model_output
        if prev_timestep > 0:
            noise = torch.randn_like(x)
            x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir + eta * posterior_std * noise
        else:
            # t == 0 (或映射到最末步) 时不加随机噪声
            x_prev = alpha_prod_t_prev**0.5 * pred_x0 
        
        # pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        # pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        # x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        **kwds):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v
            print(u.shape)
            print(v.shape)

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            # unconditional_input.input_ids = unconditional_input.input_ids[:, 1:]
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
            # predict tghe noise
            
            print("shape of model inputs:", model_inputs.shape)
            
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            latents, pred_x0 = self.step(noise_pred, t, latents, eta)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
            return image, pred_x0_list, latents_list
        return image

    @torch.no_grad()
    def __call_three_path__(
        self,
        source_prompt,
        target_prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents_s=None,
        latents_t=None,
        latents_m=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        editability_weight=1.0,  # ws parameter
        **kwds):
        """
        Three-path processing for MasaCtrl editability enhancement.
        
        Args:
            source_prompt: prompt for source reconstruction (Ls path)
            target_prompt: prompt for target editing (Lt and Lm paths)
            latents_s: initial latents for source path (Ls)
            latents_t: initial latents for target path (Lt)  
            latents_m: initial latents for editability path (Lm)
            editability_weight: ws parameter for noise update strategy
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Process prompts
        if isinstance(source_prompt, str):
            source_prompt = [source_prompt] * batch_size
        if isinstance(target_prompt, str):
            target_prompt = [target_prompt] * batch_size

        # Text embeddings for source prompt (Ls path)
        source_text_input = self.tokenizer(
            source_prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        source_text_embeddings = self.text_encoder(source_text_input.input_ids.to(DEVICE))[0]
        
        # Text embeddings for target prompt (Lt and Lm paths)
        target_text_input = self.tokenizer(
            target_prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        target_text_embeddings = self.text_encoder(target_text_input.input_ids.to(DEVICE))[0]
        
        print("source text embeddings shape:", source_text_embeddings.shape)
        print("target text embeddings shape:", target_text_embeddings.shape)

        # Define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        if latents_s is None:
            latents_s = torch.randn(latents_shape, device=DEVICE)
        if latents_t is None:
            latents_t = torch.randn(latents_shape, device=DEVICE)
        if latents_m is None:
            latents_m = torch.randn(latents_shape, device=DEVICE)

        # Unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            
            # Prepare text embeddings for three paths:
            # [uncond_s, uncond_t, uncond_m, cond_s, cond_t, cond_m]
            text_embeddings = torch.cat([
                unconditional_embeddings,  # uncond for source
                unconditional_embeddings,  # uncond for target 
                unconditional_embeddings,  # uncond for editability
                source_text_embeddings,    # cond for source
                target_text_embeddings,    # cond for target
                target_text_embeddings,    # cond for editability (uses target prompt)
            ], dim=0)
        else:
            # Without CFG: [cond_s, cond_t, cond_m]
            text_embeddings = torch.cat([
                source_text_embeddings,
                target_text_embeddings,
                target_text_embeddings,  # Lm uses target prompt
            ], dim=0)

        print("combined text embeddings shape:", text_embeddings.shape)
        print("latents_s shape:", latents_s.shape)
        print("latents_t shape:", latents_t.shape)
        print("latents_m shape:", latents_m.shape)

        # Iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        latents_s_list = [latents_s]
        latents_t_list = [latents_t]
        latents_m_list = [latents_m]
        
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Three-Path DDIM Sampler")):
            # Combine all latents for processing: [latents_s, latents_t, latents_m]
            combined_latents = torch.cat([latents_s, latents_t, latents_m], dim=0)
            
            if guidance_scale > 1.:
                # Duplicate for CFG: [latents_s, latents_t, latents_m, latents_s, latents_t, latents_m]
                model_inputs = torch.cat([combined_latents] * 2)
            else:
                model_inputs = combined_latents

            print(f"Step {i}, model_inputs shape:", model_inputs.shape)
            
            # Predict noise for all paths
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            
            if guidance_scale > 1.:
                # Split unconditional and conditional predictions
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            
            # Split noise predictions for each path
            noise_pred_s, noise_pred_t, noise_pred_m = noise_pred.chunk(3, dim=0)
            
            # Apply editability enhancement: et = em + ws * (et - em)
            # This makes the target editing move away from the editability path
            noise_pred_t_enhanced = noise_pred_m + editability_weight * (noise_pred_t - noise_pred_m)
            
            # Compute next latents for each path
            latents_s, _ = self.step(noise_pred_s, t, latents_s, eta)
            latents_t, _ = self.step(noise_pred_t_enhanced, t, latents_t, eta)
            latents_m, _ = self.step(noise_pred_m, t, latents_m, eta)
            
            latents_s_list.append(latents_s)
            latents_t_list.append(latents_t)
            latents_m_list.append(latents_m)

        # Convert final latents to images
        image_s = self.latent2image(latents_s, return_type="pt")
        image_t = self.latent2image(latents_t, return_type="pt")
        image_m = self.latent2image(latents_m, return_type="pt")
        
        if return_intermediates:
            return {
                'source': image_s,
                'target': image_t, 
                'editability': image_m,
                'latents_s_list': latents_s_list,
                'latents_t_list': latents_t_list,
                'latents_m_list': latents_m_list
            }
        
        return {
            'source': image_s,
            'target': image_t,
            'editability': image_m
        }

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents
        # print(latents)
        # exit()
        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents, start_latents
