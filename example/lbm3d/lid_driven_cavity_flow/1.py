#!/usr/bin/env python3
"""
双击即可运行：
  - 默认查看 result_000.dat
  - 弹窗显示密度、速度、压力中间切片
  - 如需改步号，改下面 STEP 变量即可
"""
import os, pickle
import numpy as np
import matplotlib.pyplot as plt

# ===== 用户只需改这里 =====
DAT_DIR = r'D:\CHen\LBM-3d\example\lid_driven_cavity_flow\Re400\results'
STEP    = 2                # 想看第几步就改几
SAVE_PNG = False          # True → 自动保存 png，不弹窗
# ==========================

def load_fields(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['t'], data['rho'], data['vel'], data['p']

def plot_middle_slice(rho, vel, p, step, save=False):
    z_mid = rho.shape[2] // 2
    rho2  = rho[:, :, z_mid]
    vel2  = vel[:, :, z_mid, :]
    speed = vel2[..., 0]
    p2    = p[:, :, z_mid]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = [f'Density (step={step})', f'Speed (step={step})', f'Pressure (step={step})']
    arrs   = [rho2, speed, p2]

    for ax, arr, ttl in zip(axes, arrs, titles):
        im = ax.imshow(arr.T, origin='lower', cmap='turbo')
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    if save:
        outpng = os.path.join(DAT_DIR, f'frame_{step:03d}.png')
        plt.savefig(outpng, dpi=500)
        print(f'Saved → {outpng}')
    else:
        plt.show()

if __name__ == '__main__':
    os.makedirs(DAT_DIR, exist_ok=True)
    target = os.path.join(DAT_DIR, f'result_{STEP:03d}.dat')
    if not os.path.isfile(target):
        print('文件不存在：', target)
        input('按回车退出...')
        exit()

    t, rho, vel, p = load_fields(target)
    print(f'已加载 step={STEP}，物理时间={t}')
    plot_middle_slice(rho, vel, p, STEP, save=SAVE_PNG)