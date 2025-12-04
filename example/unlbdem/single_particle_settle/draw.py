"""
单颗粒沉降模拟结果可视化 - 非解析版本
绘制xy平面z中间位置的速度云图，包含颗粒位置和网格线
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

# ===========================
# 参数设置（非解析案例）
# ===========================
# 域几何参数
lx = 0.1
ly = 0.16
lz = 0.1
dia = 0.015

# 非解析案例：网格尺寸固定为0.02m，大于颗粒直径
dx = 0.020  # 修改：网格尺寸

# 非解析案例：网格数计算方式不同
Nx = int(lx / dx) + 1  # 修改：+1 而非 +2
Ny = int(ly / dx) + 1
Nz = int(lz / dx) + 1

# 网格中心坐标（与模拟代码一致）
x = np.arange(Nx) * dx - 0.5 * dx
y = np.arange(Ny) * dx - 0.5 * dx
z = np.arange(Nz) * dx - 0.5 * dx

# 用于绘制contourf的网格边界坐标
# 每个网格单元从 (i*dx) 到 ((i+1)*dx)
x_edges = np.arange(Nx + 1) * dx - dx  # 网格边界
y_edges = np.arange(Ny + 1) * dx - dx

umax = 9.1e-2  # 最大速度
Re = 11.6

# 数据目录 - 修改为非解析案例的目录
data_dir = '../single_particle_settle/Re{}_un/'.format(Re)  # 修改：_un 表示 unresolved
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
            parts = line.split()
            i += 1

            if i < len(lines):
                time_particle_line = lines[i].strip().split()
                time_value = float(time_particle_line[0])
                n_particles = int(time_particle_line[1])

                i += 1
                i += 1

                if abs(time_value - timestep) < 1e-4:
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

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 12), dpi=300)
    fig.patch.set_facecolor('#F7F7F7')
    ax.set_facecolor('#F7F7F7')

    # 绘制速度云图 - 使用pcolormesh以网格单元为单位绘制
    # 裁剪到计算域内的数据
    nx_cells = int(lx / dx)
    ny_cells = int(ly / dx)
    vel_mag_clip = vel_mag[:nx_cells, :ny_cells]

    # 网格边界坐标
    x_plot = np.linspace(0, lx, nx_cells + 1)
    y_plot = np.linspace(0, ly, ny_cells + 1)

    pcm = ax.pcolormesh(x_plot, y_plot, vel_mag_clip.T,
                        cmap='plasma', vmin=0, vmax=umax, shading='flat')

    # 绘制网格线（绘制所有网格单元的边界）
    # x方向：nx_cells个网格，需要nx_cells+1条线
    nx_cells = int(lx / dx)
    ny_cells = int(ly / dx)
    '''
    for i in range(nx_cells + 1):
        xi = i * dx
        ax.axvline(x=xi, color='black', linewidth=0.5, alpha=0.85)
    for j in range(ny_cells + 1):
        yi = j * dx
        ax.axhline(y=yi, color='black', linewidth=0.5, alpha=0.85)
    '''
    # 绘制颗粒 - 非解析案例中颗粒可能小于网格
    for particle in particles:
        px, py, pz = particle['x'], particle['y'], particle['z']
        pr = particle['r']

        # 检查颗粒是否在当前z切片附近
        if abs(pz - z[z_slice]) < pr:
            dz = abs(pz - z[z_slice])
            r_effective = np.sqrt(pr**2 - dz**2) if dz < pr else 0

            if r_effective > 0:
                circle = Circle((px, py), r_effective,
                              color='white', ec='black',
                              linewidth=2, zorder=10)
                ax.add_patch(circle)


    # 添加颜色条
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical',
                        pad=0.05, aspect=30, extend='max')

    # 生成包含最大值的刻度（保留三位小数）
    ticks = np.linspace(0, umax, 6)
    cbar.set_ticks(ticks)

    # 格式化刻度标签为三位小数（例如 0.091）
    cbar.ax.set_yticklabels([f'{tick:.3f}' for tick in ticks])

    # 设置刻度字体：不加粗，仅调整大小和颜色
    cbar.ax.tick_params(labelsize=14, labelcolor='black')

    # 设置 colorbar 标题：加粗、清晰
    cbar.set_label(
        'velocity/(m/s)',
        fontsize=22,
        labelpad=10,
        fontweight='bold',
        color='black'
    )

    # 增大 colorbar 刻度标签的字体
    cbar.ax.tick_params(labelsize=18)

    ax.axis('off')
    ax.set_aspect('equal')
    # 限制显示范围为实际计算域
    ax.set_xlim([0, lx])
    ax.set_ylim([0, ly])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"已保存: {output_file}")


# ===========================
# 主程序
# ===========================
if __name__ == '__main__':
    # 打印网格信息以便调试
    print(f"网格信息: Nx={Nx}, Ny={Ny}, Nz={Nz}")
    print(f"网格尺寸: dx={dx} m")
    print(f"颗粒直径: {dia} m (颗粒/网格比 = {dia/dx:.2f})")

    if not os.path.exists(results_dir):
        print(f"错误: 找不到结果目录 {results_dir}")
        exit(1)

    p4p_file = os.path.join(data_dir, 'output.p4p')
    if not os.path.exists(p4p_file):
        print(f"错误: 找不到颗粒文件 {p4p_file}")
        exit(1)

    result_files = sorted([f for f in os.listdir(results_dir)
                          if f.startswith('result_') and f.endswith('.dat')])

    print(f"找到 {len(result_files)} 个结果文件")
    print(f"z切片位置: 索引={z_slice}, 坐标={z[z_slice]:.4f} m")
    print("开始生成速度云图...")

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