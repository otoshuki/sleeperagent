import sys
sys.path.insert(0, "ZeroNVS")

from custom_launch import main

if __name__ == "__main__":
    main(
        config_path="ZeroNVS/configs/custom_config.yaml",
        gpu="0",
        verbose=False,
        extras=[
            "data.image_path=ZeroNVS/smallmoto.png",
            "data.default_elevation_deg=31.0",
            "data.default_fovy_deg=52.55",
            "data.random_camera.fovy_range=[52.55,52.55]",
            "data.random_camera.eval_fovy_deg=52.55",
            "system.guidance.precomputed_scale=0.7",
            "trainer.max_steps=1"
        ]
    )
