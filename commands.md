python lerobot/scripts/control_robot.py \
  --robot.type=arx \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=trlc/arx_test_d405_6 \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=10 \
  --control.num_episodes=3 \
  --control.push_to_hub=false \
  --control.display_data=true

  python lerobot/scripts/control_robot.py \
  --robot.type=arx_bimanual \
  --control.type=teleoperate \
  --control.fps=30 \
  --control.display_data=true