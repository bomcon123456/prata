#! /bin/bash
export VERSION=3
export OUTPUT_DIR="/lustre/scratch/client/guardpro/trungdt21/research/face_gen/data/_processed/project_face_gen/weights/controlnet/v${VERSION}"

echo output_dir: ${OUTPUT_DIR}

accelerate launch --mixed_precision="fp16" train_controlnet.py \
	--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
	--output_dir=$OUTPUT_DIR \
	--dataset_name="/lustre/scratch/client/guardpro/trungdt21/research/face_gen/data/_processed/project_face_gen/filtered/filtered_controlnet/mediapipe" \
	--conditioning_image_column=source_img \
	--image_column=target_img \
	--caption_column=prompt \
	--resolution=512 \
	--learning_rate=5e-7 \
	--validation_image "/lustre/scratch/client/guardpro/trungdt21/research/face_gen/data/_processed/project_face_gen/filtered/filtered_controlnet/mediapipe/landmark_images/celeb/16109_yp_-62_0.png" "/lustre/scratch/client/guardpro/trungdt21/research/face_gen/data/_processed/project_face_gen/filtered/filtered_controlnet/mediapipe/landmark_images/celeb/9590_yp_48_52.png" "/lustre/scratch/client/guardpro/trungdt21/research/face_gen/data/_processed/project_face_gen/filtered/filtered_controlnet/mediapipe/landmark_images/celeb/10206_yp_56_32.png" \
	--validation_prompt "A profile portrait image of a neutral white man." "A profile portrait image of a angry asian woman." "A profile portrait image of a fear white woman." \
	--num_validation_images=8 \
	--train_batch_size=16 \
	--num_train_epochs=10 \
	--mixed_precision="fp16" \
	--tracker_project_name="mediapipe" \
	--enable_xformers_memory_efficient_attention \
	--checkpointing_steps=5000 \
	--validation_steps=5000 \
	--report_to tensorboard

# --validation_prompt "A portrait image of a neutral white man with head pose: yaw=62, pitch=0" "A portrait image of a angry asian woman with head pose: yaw=48, pitch=52" "A portrait image of a fear white woman with head pose: yaw=56, pitch=32" \
