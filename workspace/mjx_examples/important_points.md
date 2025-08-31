



1. n_frames: the number of times to step the physics pipeline for each environment step.

2. Each physics pipeline step is  set to (robot_spec.option.timestep), Here in this case its 1 / 1000 .

3. effective simulation step ( n_frames*simulation_timestep ) should be equal to the control_timestep set by "you"

4. for the stability purpuses do not change the simulation_timestep instead change the n_frames

