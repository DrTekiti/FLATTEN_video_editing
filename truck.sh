python inference.py \
--prompt "Wooden trucks drive on a racetrack." \
--neg_prompt " " \
--guidance_scale 15 \
--video_path "data/trucks-race.mp4" \
--output_path "truck/" \
--video_length 32 \
--width 512 \
--height 512 \
--frame_rate 1 \
--old_qk 1
