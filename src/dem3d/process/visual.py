'''
Visualize particle information with enhanced visual quality
'''
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.colors as mcolors

# === 配置参数 ===
input_file = r"D:\CHen\LBDEM-Taichi\example\dem3d\dem_test\output.p4p"
output_dir = "output_frames_3d"
xmin, xmax = 0, 0.2 # Domain boundaries in x-direction
ymin, ymax = 0, 0.2  # Domain boundaries in y-direction
zmin, zmax = 0, 0.6  # Domain boundaries in z-direction
radius_scale = 1
background_color = "#F8FAFC"  # 浅色背景
particle_color_hex = '#000000'  # 粒子统一颜色（可修改）
cmap_name = "plasma"  # 备用颜色映射

def hex_to_rgb(hex_color):
    """将十六进制颜色转换为RGB浮点数（0-1范围）"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)

def parse_particle_file(filename):
    timesteps = []
    current_time = None
    current_particles = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        parts = line.split()

        if len(parts) >= 2 and parts[0] in ['TIMESTEP', 'IMESTEP'] and parts[1] == 'PARTICLES':
            if current_time is not None:
                timesteps.append((current_time, current_particles))

            i += 1
            time_line = lines[i].strip().split()
            current_time = float(time_line[0])
            num_particles = int(time_line[1])
            i += 2  # 跳过标题行

            current_particles = []
            for j in range(num_particles):
                data = lines[i].strip().split()
                particle = {
                    'px': float(data[4]),
                    'py': float(data[5]),
                    'pz': float(data[6]),
                    'rad': float(data[2]) * radius_scale
                }
                current_particles.append(particle)
                i += 1
        else:
            i += 1

    if current_time is not None:
        timesteps.append((current_time, current_particles))

    return timesteps


def add_lighting_effect(ax, particles):
    """添加光照效果"""
    # 创建虚拟光源
    light_x, light_y, light_z = 0.2, 0.2, 0.8

    # 计算每个粒子的光照强度
    lighting = []
    for p in particles:
        # 简单的漫反射光照计算
        dx = light_x - p['px']
        dy = light_y - p['py']
        dz = light_z - p['pz']
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        intensity = max(0.3, 1.0 / (1 + distance))  # 最小亮度0.3
        lighting.append(intensity)

    return np.array(lighting)

def draw_3d_boundary(ax, xmin, xmax, ymin, ymax, zmin, zmax):
    """绘制3D作用域边框"""
    # 定义边框线条样式
    line_color = '#181C14'  # 边框颜色
    line_width = 2.0
    line_alpha = 0.5

    # 底面边框 (z = zmin)
    bottom_edges = [
        [[xmin, xmax], [ymin, ymin], [zmin, zmin]],  # 前边
        [[xmin, xmax], [ymax, ymax], [zmin, zmin]],  # 后边
        [[xmin, xmin], [ymin, ymax], [zmin, zmin]],  # 左边
        [[xmax, xmax], [ymin, ymax], [zmin, zmin]]   # 右边
    ]

    # 顶面边框 (z = zmax)
    top_edges = [
        [[xmin, xmax], [ymin, ymin], [zmax, zmax]],  # 前边
        [[xmin, xmax], [ymax, ymax], [zmax, zmax]],  # 后边
        [[xmin, xmin], [ymin, ymax], [zmax, zmax]],  # 左边
        [[xmax, xmax], [ymin, ymax], [zmax, zmax]]   # 右边
    ]

    # 垂直边框 (连接底面和顶面)
    vertical_edges = [
        [[xmin, xmin], [ymin, ymin], [zmin, zmax]],  # 前左
        [[xmax, xmax], [ymin, ymin], [zmin, zmax]],  # 前右
        [[xmin, xmin], [ymax, ymax], [zmin, zmax]],  # 后左
        [[xmax, xmax], [ymax, ymax], [zmin, zmax]]   # 后右
    ]

    # 绘制所有边框线
    all_edges = bottom_edges + top_edges + vertical_edges

    for edge in all_edges:
        ax.plot(edge[0], edge[1], edge[2],
                color=line_color,
                linewidth=line_width,
                alpha=line_alpha,
                linestyle='-')

    # 添加角点高亮
    corners_x = [xmin, xmax, xmin, xmax, xmin, xmax, xmin, xmax]
    corners_y = [ymin, ymin, ymax, ymax, ymin, ymin, ymax, ymax]
    corners_z = [zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax]

    ax.scatter(corners_x, corners_y, corners_z,
               c='#F8FAFC',  # 角点颜色
               s=150,
               alpha=0.8,
               edgecolors='white',
               linewidths=0.6)

def visualize_timesteps(timesteps):
    os.makedirs(output_dir, exist_ok=True)

    # 将粒子颜色转换为RGB数值格式
    particle_rgb = hex_to_rgb(particle_color_hex)

    for step, (time, particles) in enumerate(timesteps):
        fig = plt.figure(figsize=(16, 12), dpi=600)
        fig.patch.set_facecolor(background_color)

        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(background_color)

        # 提取坐标与半径
        px = np.array([p['px'] for p in particles])
        py = np.array([p['py'] for p in particles])
        pz = np.array([p['pz'] for p in particles])
        radii = np.array([p['rad'] for p in particles])

        # 生成统一颜色列表（数值格式）
        colors = [list(particle_rgb) for _ in particles]

        # 计算光照并应用到颜色
        lighting = add_lighting_effect(ax, particles)
        for i in range(len(colors)):
            # 对RGB三个通道分别应用光照强度
            colors[i][0] *= lighting[i]  # 红色通道
            colors[i][1] *= lighting[i]  # 绿色通道
            colors[i][2] *= lighting[i]  # 蓝色通道

        # 绘制粒子主体
        scatter = ax.scatter(
            px, py, pz,
            s=(radii * 2000) ** 2,
            c=colors,  # 使用数值格式的颜色列表
            alpha=0.75,
            edgecolors='white',
            linewidths=0.5,
            depthshade=True
        )

        # 添加粒子边缘高光效果
        ax.scatter(
            px, py, pz,
            s=(radii * 1200) ** 2,
            c='white',
            alpha=0.15,
            edgecolors='none'
        )

        # 添加3D作用域边框
        draw_3d_boundary(ax, xmin, xmax, ymin, ymax, zmin, zmax)

        # 设置范围与比例
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_box_aspect([xmax - xmin, ymax - ymin, zmax - zmin])

        # 美化坐标轴
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # 设置坐标轴线条
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)

        # 隐藏坐标轴标签和刻度
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # 设置视角
        ax.view_init(elev=10, azim=-10)

        # 添加标题
        #ax.text2D(0.02, 0.98, f"Time: {time:.3f}s",
                 #transform=ax.transAxes,
                 #fontsize=12,
                 #color='black',  # 标题颜色（适配浅色背景）
                 #weight='bold',
                 #bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        # 保存图片
        output_path = os.path.join(output_dir, f"frame_{step:04d}.png")
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1,
            facecolor=fig.get_facecolor(),
            edgecolor='none',
            format='png'
        )
        plt.close()

        # 进度提示
        if step % 10 == 0:
            print(f"Generated frame {step}/{len(timesteps)}")

def create_summary_statistics(timesteps):
    """创建粒子统计信息"""
    print("=== 粒子系统统计 ===")
    print(f"总时间步数: {len(timesteps)}")
    if timesteps:
        avg_particles = np.mean([len(particles) for _, particles in timesteps])
        print(f"平均粒子数: {avg_particles:.1f}")

        # 分析粒子分布
        all_z = []
        all_radii = []
        for _, particles in timesteps:
            for p in particles:
                all_z.append(p['pz'])
                all_radii.append(p['rad'])

        print(f"Z坐标范围: {min(all_z):.3f} 到 {max(all_z):.3f}")
        print(f"粒子半径范围: {min(all_radii):.3f} 到 {max(all_radii):.3f}")

if __name__ == "__main__":
    try:
        timesteps = parse_particle_file(input_file)
        create_summary_statistics(timesteps)
        visualize_timesteps(timesteps)
        print(f"\n✅ 成功生成 {len(timesteps)} 个高清3D帧图像")
        print(f"📁 输出目录: '{output_dir}'")
        print(f"🎨 图像规格: 16x12英寸, 400 DPI")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{input_file}'")
        print("请检查文件路径是否正确")
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")