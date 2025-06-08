import sys
import os

sys.path.append(os.path.abspath("ZeroNVS"))

from zero123gen import Zero123Generator

if __name__ == "__main__":
    device = "cuda"
    config_path = "ZeroNVS/zeronvs_config.yaml"
    ckpt_path = "ZeroNVS/zeronvs.ckpt"
    image_path = "ZeroNVS/smallmoto.png"
    precomputed_scale = 0.7
    generator = Zero123Generator(config_path, ckpt_path, device, precomputed_scale)

    latents = generator.generate_latents(image_path, azimuths_deg=[20, -20], scale=7.5, ddim_steps=20)
    images = generator.generate_views_from_latents(latents)
    for i, img in enumerate(images):
        img.save(f"output_view_{i}.png")
