source venv/bin/activate
python3 convert_diffusers_to_original_sdxl.py \
	--model_path="/mnt/storage/projects/sdxl-train/output/34_orginal_size_upscale_sdxl_train_34.py_gpus3_bsz11_gradAccum6_lr5e-07_res1024/10/" \
	--checkpoint_path=//mnt/storage/projects/sdxl-train/output/34_orginal_size_upscale_sdxl_train_34.py_gpus3_bsz11_gradAccum6_lr5e-07_res1024/10/output.safetensors \
	--half \
	--use_safetensors
