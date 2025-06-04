python lerobot/scripts/control_robot.py \
  --robot.type=arx_bimanual \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Clean." \
  --control.warmup_time_s=3 \
  --control.episode_time_s=600 \
  --control.reset_time_s=30 \
  --control.num_episodes=3 \
  --control.push_to_hub=false \
  --control.display_data=false \
  --control.repo_id=trlc/clean_dishes


python lerobot/scripts/control_robot.py \
  --robot.type=arx_bimanual \
  --control.type=teleoperate \
  --control.fps=30 \
  --control.display_data=true

sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1




python lerobot/scripts/control_robot.py \
  --robot.type=arx \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.warmup_time_s=5 \
  --control.episode_time_s=120 \
  --control.reset_time_s=30 \
  --control.num_episodes=3 \
  --control.push_to_hub=false \
  --control.policy.path=/home/ubuntu/trlc/lerobot/outputs/train/act_lego_block_annotated/100000/pretrained_model \
  --control.display_data=true \
  --control.repo_id=trlc/eval_masking9w
