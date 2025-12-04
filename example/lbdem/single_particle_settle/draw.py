"""
单颗粒沉降模拟结果可视化
绘制xy平面z中间位置的速度云图，包含颗粒位置和网格线
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

# ===========================
# 参数设置（需要与模拟参数一致）
# ===========================
# 域几何参数
lx = 0.1
ly = 0.16
lz = 0.1
dia = 0.015
dx = dia / 16
Nx = int(lx / dx) + 2
Ny = int(ly / dx) + 2
Nz = int(lz / dx) + 2

x = np.arange(Nx) * dx - 0.5 * dx
y = np.arange(Ny) * dx - 0.5 * dx
z = np.arange(Nz) * dx - 0.5 * dx

umax = 9.1e-2  # 最大速度
Re = 11.6

# 数据目录
data_dir = '../single_particle_settle/Re{}_fully/'.format(Re)
results_dir = os.path.join(data_dir, 'results')
output_dir = os.path.join(data_dir, 'figures')
os.makedirs(output_dir, exist_ok=True)

# z切片位置（中间位置）
z_slice = Nz // 2

# ===========================
# 读取和可视化函数
# ===========================
def load_result(file_path):
    """加载模拟结果数据"""
    with open(file_path, 'rb') as fid:
        data = pickle.load(fid)
    return data


def parse_p4p_for_timestep(p4p_file, timestep):
    """从p4p文件中读取特定时间步的颗粒位置"""
    particles = []
    with open(p4p_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('TIMESTEP'):
            # 格式: TIMESTEP  PARTICLES
            #       time_value  n_particles
            # 例如: "0.04994612068965516 1"
            parts = line.split()
            # parts[0] = 'TIMESTEP', parts[1] = 'PARTICLES'
            i += 1  # 移到下一行（时间和颗粒数）

            if i < len(lines):
                time_particle_line = lines[i].strip().split()
                time_value = float(time_particle_line[0])
                n_particles = int(time_particle_line[1])

                # 跳过标题行 "ID  GROUP  RAD  MASS  PX  PY  PZ  VX  VY  VZ"
                i += 1
                i += 1  # 现在指向第一个颗粒数据行

                if abs(time_value - timestep) < 1e-4:
                    # 找到对应时间步，读取所有颗粒
                    particle_data = []
                    for _ in range(n_particles):
                        if i < len(lines):
                            parts = lines[i].strip().split()
                            if len(parts) >= 7:
                                particle_data.append({
                                    'id': int(parts[0]),
                                    'group': int(parts[1]),
                                    'r': float(parts[2]),
                                    'mass': float(parts[3]),
                                    'x': float(parts[4]),
                                    'y': float(parts[5]),
                                    'z': float(parts[6])
                                })
                            i += 1
                    return particle_data
                else:
                    # 跳过当前时间步的颗粒数据
                    i += n_particles
        else:
            i += 1

    return []


def plot_velocity_field(result_file, p4p_file, output_file):
    """绘制速度云图"""
    # 加载流场数据
    data = load_result(result_file)
    t = data['t']
    velf = data['velf']  # 速度场 [Nx, Ny, Nz, 3]

    # 提取z切片的速度场
    vel_slice = velf[:, :, z_slice, :]

    # 计算速度大小
    vel_mag = np.sqrt(vel_slice[:, :, 0]**2 +
                      vel_slice[:, :, 1]**2 +
                      vel_slice[:, :, 2]**2)

    # 读取颗粒位置
    particles = parse_p4p_for_timestep(p4p_file, t)

    # 创建网格坐标
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 12), dpi=300)
    fig.patch.set_facecolor('#F7F7F7')
    ax.set_facecolor('#F7F7F7')

    # 绘制速度云图
    levels = np.linspace(0, umax, 50)
    contourf = ax.contourf(X, Y, vel_mag, levels=levels,
                           cmap='plasma', extend='both')

    # 绘制网格线（每个格子对应一个网格单元 dx）

    # 绘制颗粒
    for particle in particles:
        px, py, pz = particle['x'], particle['y'], particle['z']
        pr = particle['r']

        # 检查颗粒是否在当前z切片附近
        if abs(pz - z[z_slice]) < pr:
            # 计算在切片上的有效半径
            dz = abs(pz - z[z_slice])
            r_effective = np.sqrt(pr ** 2 - dz ** 2) if dz < pr else 0

            if r_effective > 0:
                circle = Circle((px, py), r_effective,
                                color='white', ec='black',
                                linewidth=2, zorder=10)
                ax.add_patch(circle)

    # === 新的 colorbar 设置（按要求） ===
    cbar = plt.colorbar(contourf, ax=ax, orientation='vertical',
                        pad=0.05, aspect=30, extend='max')

    ticks = np.linspace(0, umax, 6)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f'{tick:.3f}' for tick in ticks])
    cbar.ax.tick_params(labelsize=14, labelcolor='black')

    cbar.set_label(
        'velocity/(m/s)',
        fontsize=16,
        labelpad=15,
        fontweight='bold',
        color='black'
    )
    # === colorbar 设置结束 ===

    # 关闭坐标轴和标题
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


# ===========================
# 主程序
# ===========================
if __name__ == '__main__':
    # 检查数据目录是否存在
    if not os.path.exists(results_dir):
        print(f"错误: 找不到结果目录 {results_dir}")
        exit(1)

    p4p_file = os.path.join(data_dir, 'output.p4p')
    if not os.path.exists(p4p_file):
        print(f"错误: 找不到颗粒文件 {p4p_file}")
        exit(1)

    # 获取所有结果文件
    result_files = sorted([f for f in os.listdir(results_dir)
                          if f.startswith('result_') and f.endswith('.dat')])

    print(f"找到 {len(result_files)} 个结果文件")
    print(f"z切片位置: 索引={z_slice}, 坐标={z[z_slice]:.4f} m")
    print("开始生成速度云图...")

    # 处理每个时间步
    for result_file in result_files:
        result_path = os.path.join(results_dir, result_file)
        output_file = os.path.join(output_dir,
                                   result_file.replace('.dat', '.png'))

        try:
            plot_velocity_field(result_path, p4p_file, output_file)
        except Exception as e:
            print(f"处理 {result_file} 时出错: {e}")

    print(f"\n所有图像已保存到: {output_dir}")
    print("可视化完成！")