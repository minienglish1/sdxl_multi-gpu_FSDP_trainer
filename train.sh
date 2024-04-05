#FSDP
#2 gpus adam8bit 1024 resolution batch size 10

#additonal commands
#	--cached_dataset_lists None \ #use .list instead if cached dir, not tested
#	--output_dir \ 	#if not used, script will create output dir name base on training parameters
#	--pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \ #change model to your base model
#	--save_state \ #save unet when saving pipeline
#	--load_saved_state \ #dir where unet is saved
#	--weight_dtype torch.float16 \ #script is 100% designed for fp16, don't change this unless you know what you're doing
#	--polynomial_lr_end 1e-8 \ #lr_end for polynominal lr scheduler
#	--polynomial_power 1 \ #polynominal lr scheduler power
#	--upscale_use_GFPGAN \	#use GFPGAN when upscaling, useful for photos, not useful for non-photos
#	--save_upscale_samples \ #saves org & upscaled images, useful to check if upscaling meets your quality standard
#	--save_samples \ #save sample_images during training
#	--validation_image #use validation_image (IS/FID/KID/LPIPS/HPSv2) scoring
#	--verify_cached_dataset_hash_values #verify cached dataset integrity before training
#	--validation_loss #use validation_loss
#	--load_saved_state \ #load saved unet

#see sdxl_FSDP_train for full list of arguments


#export TRANSFORMERS_OFFLINE=1 #uncomment to train offline
source venv/bin/activate
accelerate launch \
	--config_file FSDP_12.yaml \
	sdxl_FSDP_train_48.py \
	--project_name 48_sample_dataset\
	--pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
	--cached_dataset_dirs cache \
	--max_resolution 1024 \
	--min_resolution 256 \
	--upscale_to_resolution 1024 \
	--set_seed 123 \
	--train_batch_size 10 \
	--num_train_epochs 10 \
	--gradient_accumulation_steps 5 \
	--conditional_dropout_percent 0.1 \
	--learning_rate 1e-5 \
	--learning_rate_scheduler "constant_with_warmup" \
	--percent_lr_warm_up_steps 0.01 \
	--start_save_model_epoch 0 \
	--save_model_every_n_epochs 5 \
	--save_state \
	--save_samples \
	--start_save_samples_epoch 0 \
	--save_samples_every_n_epochs 1 \
	--num_sample_images 14 \
	--validation_loss \
	--validation_loss_percent 0.10 \
	--validation_loss_every_n_epochs 1 \
	--validation_image \
	--validation_image_percent 0.10 \
	--validation_image_every_n_epochs 1
