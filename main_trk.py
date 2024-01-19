import numpy as np
import matplotlib.pyplot as plt
import pykitti.pykitti as pykitti
import sys

np.random.seed(777)
sys.path.append('./src')

from kalman_filters import ExtendedKalmanFilter as EKF
from utils import lla_to_enu, normalize_angles


oxt_dir = "/home/apera/mhmd/kittiMOT/data_kittiMOT/data_tracking_oxts/training/oxts/0000.txt"
image_dir = "/home/apera/mhmd/kittiMOT/data_kittiMOT/data_tracking_image_2/training/image_02/0000"

dataset = pykitti.tracking("/home/apera/mhmd/kittiMOT/data_kittiMOT/training", "0001")


gt_trajectory_lla = []  # [longitude(deg), latitude(deg), altitude(meter)] x N
gt_yaws = []  # [yaw_angle(rad),] x N
gt_yaw_rates= []  # [vehicle_yaw_rate(rad/s),] x N
gt_forward_velocities = []  # [vehicle_forward_velocity(m/s),] x N
for oxts_data in dataset.oxts:
    packet = oxts_data.packet
    gt_trajectory_lla.append([
        packet.lon,
        packet.lat,
        packet.alt
    ])
    gt_yaws.append(packet.yaw)
    gt_yaw_rates.append(packet.wz)
    gt_forward_velocities.append(packet.vf)

gt_trajectory_lla = np.array(gt_trajectory_lla).T
gt_yaws = np.array(gt_yaws)
gt_yaw_rates = np.array(gt_yaw_rates)
gt_forward_velocities = np.array(gt_forward_velocities)

lons, lats, _ = gt_trajectory_lla

# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.plot(lons, lats)
# ax.set_xlabel('longitude [deg]')
# ax.set_ylabel('latitude [deg]')
# ax.grid()

# origin = gt_trajectory_lla[:, 0]  # set the initial position to the origin
# gt_trajectory_xyz = lla_to_enu(gt_trajectory_lla, origin)

# xs, ys, _ = gt_trajectory_xyz
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.plot(xs, ys)
# ax.set_xlabel('X [m]')
# ax.set_ylabel('Y [m]')
# ax.grid()

ts = dataset.timestamps

time_diffs = np.diff(ts)
print(1/np.average(time_diffs), np.std(time_diffs))
# plt.figure()
# plt.scatter(ts[1:], time_diffs)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].plot(ts, gt_yaws)
ax[0, 0].set_xlabel('time elapsed [sec]')
ax[0, 0].set_ylabel('ground-truth yaw angle [rad]')

ax[1, 0].plot(ts, gt_yaw_rates)
ax[1, 0].set_xlabel('time elapsed [sec]')
ax[1, 0].set_ylabel('ground-truth yaw rate [rad/s]')

ax[1, 1].plot(ts, gt_forward_velocities)
ax[1, 1].set_xlabel('time elapsed [sec]')
ax[1, 1].set_ylabel('ground-truth forward velocity [m/s]')

plt.show()