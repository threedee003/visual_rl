import gymnasium as gym
import mani_skill.envs
import numpy as np
import matplotlib.pyplot as plt



env = gym.make(
    "PickCube-v1",
    num_envs=2,
    obs_mode="rgbd", 
    control_mode="pd_ee_delta_pose", 
    render_mode="human",
    sensor_configs=dict(width=640, height=480)
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=41)
# # done = False
# for i in range(100000):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()  
# env.close()


img = obs['sensor_data']['base_camera']['rgb']
depth = obs['sensor_data']['base_camera']['depth']
print(img.shape)
print(img.dtype)
print(depth.shape)
print(depth.dtype)


'''
output for the code to understand maniskill observation and action space.


Observation space Dict('agent': Dict('qpos': Box(-inf, inf, (1, 9), float32), 'qvel': Box(-inf, inf, (1, 9), float32)), 'extra': Dict('is_grasped': Box(False, True, (1,), bool), 'tcp_pose': Box(-inf, inf, (1, 7), float32), 'goal_pos': Box(-inf, inf, (1, 3), float32)), 'sensor_param': Dict('base_camera': Dict('extrinsic_cv': Box(-inf, inf, (1, 3, 4), float32), 'cam2world_gl': Box(-inf, inf, (1, 4, 4), float32), 'intrinsic_cv': Box(-inf, inf, (1, 3, 3), float32))), 'sensor_data': Dict('base_camera': Dict('depth': Box(-32768, 32767, (1, 480, 640, 1), int16), 'rgb': Box(0, 255, (1, 480, 640, 3), uint8))))
Action space Box(-1.0, 1.0, (7,), float32)
torch.Size([1, 480, 640, 3])
torch.uint8
torch.Size([1, 480, 640, 1])
torch.int16



'''





# img = img.squeeze()
# depth = depth.squeeze()

# # print(depth.shape)
# plt.imsave('test.png', img.cpu().numpy())
# plt.imsave('test_depth.png', depth.cpu().numpy())
# plt.imshow(img.cpu().numpy())
# plt.show()
