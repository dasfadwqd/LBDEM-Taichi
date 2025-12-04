#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postprocess_dat.py
读取 LBM-DEM 模拟生成的 result_xxx.dat（pickle），
提取 Fx、Tz 并画图，同时输出 csv 文件。
"""
import os
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 找到所有 dat 文件（支持通配符）
res_dir = r'D:\CHen\LBM-DEM-3dn\example\unlbdem\single_particle_settle\Re31.9_un\results'
dat_list = sorted(glob.glob(os.path.join(res_dir, 'result_*.dat')))
if not dat_list:
    raise FileNotFoundError(f'在 {res_dir} 下未找到 result_*.dat 文件')

# 2. 批量读取
t_lst, fy_lst = [], []
for f in dat_list:
    with open(f, 'rb') as handle:
        data = pickle.load(handle)
    t_lst.append(data['t'])
    fy_lst.append(data['Fy'])


# 3. 转成 numpy 数组
t = np.array(t_lst)
fy = np.array(fy_lst)


# 4. 保存 csv
df = pd.DataFrame({'time(s)': t, 'Fy(N)': fy})
csv_path = os.path.join(res_dir, 'Fx_Tz_history.csv')
df.to_csv(csv_path, index=False, float_format='%.6e')
print(f'数据已保存 -> {csv_path}')

# 5. 画图
plt.figure(figsize=(6, 4))
plt.plot(t, fy, label='Fy')
plt.xlabel('time (s)')
plt.ylabel('force ')
plt.title('Fy history')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(res_dir, 'Fx_Tz_history.png'), dpi=300)
plt.show()