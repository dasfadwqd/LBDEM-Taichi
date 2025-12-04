import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# 检查可选依赖
try:
    import openpyxl

    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("⚠️ Warning: openpyxl not found. Excel export will be skipped.")

try:
    from scipy.interpolate import griddata

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️ Warning: scipy not found. High-resolution streamlines will be skipped.")


class CavityFlowAnalyzer:
    """3D顶盖驱动空腔流动分析器"""

    def __init__(self, data_dir, Re=400):
        self.data_dir = data_dir
        self.Re = Re
        self.results_dir = os.path.join(data_dir, 'results')
        self.analysis_dir = os.path.join(data_dir, 'analysis')

        # 创建分析输出目录
        os.makedirs(self.analysis_dir, exist_ok=True)

        self.time_series = []
        self.velocity_data = []
        self.density_data = []
        self.pressure_data = []

        # 流体力学常用的色彩映射
        self.colormaps = {
            'velocity': 'viridis',
            'pressure': 'RdBu_r',
            'density': 'plasma',
            'vorticity': 'seismic',
            'temperature': 'hot'
        }

        # 自定义CFD风格色彩映射
        self._setup_custom_colormaps()

    def _setup_custom_colormaps(self):
        """设置自定义色彩映射"""
        # CFD软件风格的速度色彩映射
        cfd_colors = ['#000033', '#000055', '#0000ff', '#0055ff',
                      '#00ffff', '#55ff00', '#ffff00', '#ff5500', '#ff0000']
        self.cfd_cmap = LinearSegmentedColormap.from_list('cfd_velocity', cfd_colors)

        # 压力场色彩映射（蓝白红）
        pressure_colors = ['#0000ff', '#4444ff', '#8888ff', '#ccccff',
                           '#ffffff', '#ffcccc', '#ff8888', '#ff4444', '#ff0000']
        self.pressure_cmap = LinearSegmentedColormap.from_list('cfd_pressure', pressure_colors)

        # 密度场色彩映射（黑体辐射风格）
        density_colors = ['#000000', '#440000', '#880000', '#cc0000',
                          '#ff0000', '#ff4400', '#ff8800', '#ffcc00', '#ffff00']
        self.density_cmap = LinearSegmentedColormap.from_list('cfd_density', density_colors)

        # 涡度场色彩映射（对称蓝白红）
        vorticity_colors = ['#0000ff', '#4444ff', '#8888ff', '#ccccff',
                            '#ffffff', '#ffcccc', '#ff8888', '#ff4444', '#ff0000']
        self.vorticity_cmap = LinearSegmentedColormap.from_list('cfd_vorticity', vorticity_colors)

        # 更新色彩映射字典
        self.colormaps.update({
            'cfd_velocity': self.cfd_cmap,
            'cfd_pressure': self.pressure_cmap,
            'cfd_density': self.density_cmap,
            'cfd_vorticity': self.vorticity_cmap
        })

    def load_simulation_data(self):
        """加载仿真数据"""
        print("📂 加载仿真数据...")

        result_files = [f for f in os.listdir(self.results_dir) if f.endswith('.dat')]
        result_files.sort()

        for i, filename in enumerate(result_files):
            try:
                with open(os.path.join(self.results_dir, filename), 'rb') as f:
                    data = pickle.load(f)

                self.time_series.append(data['t'])
                self.velocity_data.append(data['vel'])
                self.density_data.append(data['rho'])
                self.pressure_data.append(data['p'])

                if i % 1 == 0:
                    print(f"   已加载: {filename} (t={data['t']:.3f}s)")

            except Exception as e:
                print(f"❌ 加载文件 {filename} 失败: {e}")

        print(f"✅ 共加载 {len(self.time_series)} 个时间步数据")

    def analyze_convergence(self):
        """分析收敛性"""
        print("📊 分析收敛性...")

        # 计算速度场的变化
        velocity_norms = []
        velocity_changes = []

        for i, vel in enumerate(self.velocity_data):
            # 计算速度场的范数
            vel_norm = np.sqrt(np.mean(vel ** 2))
            velocity_norms.append(vel_norm)

            if i > 0:
                # 计算相对于前一时间步的变化
                vel_change = np.sqrt(np.mean((vel - self.velocity_data[i - 1]) ** 2))
                velocity_changes.append(vel_change / velocity_norms[i - 1])

        # 绘制收敛图
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.semilogy(self.time_series, velocity_norms)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity Field Norm')
        plt.title('Velocity Field Evolution')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        if velocity_changes:
            plt.semilogy(self.time_series[1:], velocity_changes)
            plt.xlabel('Time [s]')
            plt.ylabel('Relative Change')
            plt.title('Convergence Rate')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'convergence_analysis.png'), dpi=300)
        plt.close()

        return velocity_norms, velocity_changes

    def analyze_flow_features(self):
        """分析流动特征"""
        print("🌪️ 分析流动特征...")

        # 使用最后时间步的数据
        final_vel = self.velocity_data[-1]
        final_rho = self.density_data[-1]

        Nx, Ny, Nz = final_vel.shape[:3]

        # 计算速度大小
        vel_magnitude = np.sqrt(final_vel[:, :, :, 0] ** 2 +
                                final_vel[:, :, :, 1] ** 2 +
                                final_vel[:, :, :, 2] ** 2)

        # 计算涡度 (简化为中心切片的2D涡度)
        z_mid = Nz // 2
        u = final_vel[:, :, z_mid, 0]
        v = final_vel[:, :, z_mid, 1]

        # 计算涡度 (数值微分)
        vorticity = np.zeros((Nx - 2, Ny - 2))
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                dvdx = (v[i + 1, j] - v[i - 1, j]) / 2.0
                dudy = (u[i, j + 1] - u[i, j - 1]) / 2.0
                vorticity[i - 1, j - 1] = dvdx - dudy

        # 分析统计量
        flow_stats = {
            'max_velocity': np.max(vel_magnitude),
            'mean_velocity': np.mean(vel_magnitude),
            'max_vorticity': np.max(np.abs(vorticity)),
            'velocity_std': np.std(vel_magnitude),
            'max_u_x': np.max(final_vel[:, :, :, 0]),
            'min_u_x': np.min(final_vel[:, :, :, 0]),
            'max_u_y': np.max(final_vel[:, :, :, 1]),
            'min_u_y': np.min(final_vel[:, :, :, 1])
        }

        # 保存统计信息
        with open(os.path.join(self.analysis_dir, 'flow_statistics.txt'), 'w') as f:
            f.write(f"3D Lid-Driven Cavity Flow Analysis (Re={self.Re})\n")
            f.write("=" * 50 + "\n")
            for key, value in flow_stats.items():
                f.write(f"{key:20s}: {value:.6e}\n")

        return flow_stats, vorticity

    def create_slice_visualizations(self):
        """创建切片可视化 - 移除所有坐标轴元素"""
        print("🖼️ 创建切片可视化...")

        final_vel = self.velocity_data[-1]
        final_rho = self.density_data[-1]
        final_p = self.pressure_data[-1]

        Nx, Ny, Nz = final_vel.shape[:3]

        # 速度大小
        vel_magnitude = np.sqrt(final_vel[:, :, :, 0] ** 2 +
                                final_vel[:, :, :, 1] ** 2 +
                                final_vel[:, :, :, 2] ** 2)

        # 创建不同切片的可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # XY切片 (Z方向中心)
        z_mid = Nz // 2
        xy_vel = vel_magnitude[:, :, z_mid]
        xy_u = final_vel[:, :, z_mid, 0]
        xy_v = final_vel[:, :, z_mid, 1]
        xy_p = final_p[:, :, z_mid]

        # 速度场 - 使用CFD风格色彩映射
        im1 = axes[0, 0].imshow(xy_vel.T, origin='lower', cmap=self.cfd_cmap)
        axes[0, 0].set_title(f'XY - Velocity[m/s]', fontsize=12, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        cbar1.set_label('Velocity [m/s]', fontsize=10)

        # 添加流线
        x_coords = np.arange(Nx)
        y_coords = np.arange(Ny)
        X, Y = np.meshgrid(x_coords, y_coords)
        axes[0, 0].streamplot(X, Y, xy_u.T, xy_v.T, density=1.5, color='white', linewidth=0.8, arrowsize=1.2)
        # 移除坐标标签
        # axes[0, 0].set_xlabel('X', fontsize=10)
        # axes[0, 0].set_ylabel('Y', fontsize=10)
        # 隐藏刻度和边框
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        for spine in axes[0, 0].spines.values():
            spine.set_visible(False)

        # XZ切片 (Y方向中心)
        y_mid = Ny // 2
        xz_vel = vel_magnitude[:, y_mid, :]

        im2 = axes[0, 1].imshow(xz_vel.T, origin='lower', cmap=self.cfd_cmap)
        axes[0, 1].set_title(f'XZ - Velocity[m/s]', fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        cbar2.set_label('Velocity [m/s]', fontsize=10)
        # 移除坐标标签
        # axes[0, 1].set_xlabel('X', fontsize=10)
        # axes[0, 1].set_ylabel('Z', fontsize=10)
        # 隐藏刻度和边框
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        for spine in axes[0, 1].spines.values():
            spine.set_visible(False)

        # YZ切片 (X方向中心)
        x_mid = Nx // 2
        yz_vel = vel_magnitude[x_mid, :, :]

        im3 = axes[0, 2].imshow(yz_vel.T, origin='lower', cmap=self.cfd_cmap)
        axes[0, 2].set_title(f'YZ - Velocity[m/s]', fontsize=12, fontweight='bold')
        cbar3 = plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
        cbar3.set_label('Velocity [m/s]', fontsize=10)
        # 移除坐标标签
        # axes[0, 2].set_xlabel('Y', fontsize=10)
        # axes[0, 2].set_ylabel('Z', fontsize=10)
        # 隐藏刻度和边框
        axes[0, 2].set_xticks([])
        axes[0, 2].set_yticks([])
        for spine in axes[0, 2].spines.values():
            spine.set_visible(False)

        # 压力场切片 - 使用压力色彩映射
        im4 = axes[1, 0].imshow(xy_p.T, origin='lower', cmap=self.pressure_cmap)
        axes[1, 0].set_title('XY - Pressure [Pa]', fontsize=12, fontweight='bold')
        cbar4 = plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
        cbar4.set_label('Pressure [Pa]', fontsize=10)
        # 移除坐标标签
        # axes[1, 0].set_xlabel('X', fontsize=10)
        # axes[1, 0].set_ylabel('Y', fontsize=10)
        # 隐藏刻度和边框
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        for spine in axes[1, 0].spines.values():
            spine.set_visible(False)

        # 密度场切片 - 使用密度色彩映射
        im5 = axes[1, 1].imshow(final_rho[:, y_mid, :].T, origin='lower', cmap=self.density_cmap)
        axes[1, 1].set_title('XZ - Density [kg/m³]', fontsize=12, fontweight='bold')
        cbar5 = plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
        cbar5.set_label('Density [kg/m³]', fontsize=10)
        # 移除坐标标签
        # axes[1, 1].set_xlabel('X', fontsize=10)
        # axes[1, 1].set_ylabel('Z', fontsize=10)
        # 隐藏刻度和边框
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        for spine in axes[1, 1].spines.values():
            spine.set_visible(False)

        # 速度矢量场
        # 下采样矢量以避免过于密集
        skip = 3
        X_sub = X[::skip, ::skip]
        Y_sub = Y[::skip, ::skip]
        U_sub = xy_u[::skip, ::skip]
        V_sub = xy_v[::skip, ::skip]
        vel_mag_sub = xy_vel[::skip, ::skip]

        # 使用速度大小作为颜色映射
        quiver = axes[1, 2].quiver(X_sub, Y_sub, U_sub.T, V_sub.T, vel_mag_sub.T,
                                   cmap=self.cfd_cmap, scale=None, alpha=0.8)
        axes[1, 2].set_title('XY - Velocity Vectors', fontsize=12, fontweight='bold')
        cbar6 = plt.colorbar(quiver, ax=axes[1, 2], shrink=0.8)
        cbar6.set_label('Velocity [m/s]', fontsize=10)
        # 移除坐标标签
        # axes[1, 2].set_xlabel('X', fontsize=10)
        # axes[1, 2].set_ylabel('Y', fontsize=10)
        axes[1, 2].set_aspect('equal')
        # 隐藏刻度和边框
        axes[1, 2].set_xticks([])
        axes[1, 2].set_yticks([])
        for spine in axes[1, 2].spines.values():
            spine.set_visible(False)

        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(self.analysis_dir, 'slice_visualization.png'), dpi=600, bbox_inches='tight')
        plt.close()


    def create_3d_visualization(self):
        """创建3D可视化"""
        print("🎯 创建3D可视化...")

        final_vel = self.velocity_data[-1]
        Nx, Ny, Nz = final_vel.shape[:3]

        # 创建3D速度矢量图 (下采样以提高性能)
        step = 4  # 下采样步长

        x = np.arange(0, Nx, step)
        y = np.arange(0, Ny, step)
        z = np.arange(0, Nz, step)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        U = final_vel[::step, ::step, ::step, 0]
        V = final_vel[::step, ::step, ::step, 1]
        W = final_vel[::step, ::step, ::step, 2]

        # 计算速度大小用于颜色映射
        vel_mag = np.sqrt(U ** 2 + V ** 2 + W ** 2)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制3D矢量场
        q = ax.quiver(X, Y, Z, U, V, W,
                      length=2.0, normalize=True,
                      cmap='viridis', alpha=0.6)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_title(f'3D Velocity Field (Re={self.Re})')

        plt.colorbar(q, ax=ax, shrink=0.5, aspect=20)

        plt.savefig(os.path.join(self.analysis_dir, '3d_velocity_field.png'), dpi=600)
        plt.close()

    def analyze_center_line_profiles(self):
        """分析中心线速度分布 - 增强版本，包括数据导出和流线图"""
        print("📏 分析中心线速度分布...")

        final_vel = self.velocity_data[-1]
        Nx, Ny, Nz = final_vel.shape[:3]

        # 垂直中心线 (x=Nx/2, z=Nz/2)
        x_mid, z_mid = Nx // 2, Nz // 2
        y_coords = np.arange(Ny)
        vertical_u = final_vel[x_mid, :, z_mid, 0]  # U速度分量
        vertical_v = final_vel[x_mid, :, z_mid, 1]  # V速度分量

        # 水平中心线 (y=Ny/2, z=Nz/2)
        y_mid = Ny // 2
        x_coords = np.arange(Nx)
        horizontal_u = final_vel[:, y_mid, z_mid, 0]  # U速度分量
        horizontal_v = final_vel[:, y_mid, z_mid, 1]  # V速度分量

        # ===== 数据导出功能 =====
        self._export_centerline_data(y_coords, vertical_u, vertical_v,
                                     x_coords, horizontal_u, horizontal_v)

        # ===== 创建流线图 =====
        self._create_streamline_plot(final_vel, z_mid)

        # ===== 创建原有的中心线分析图 =====
        self._create_centerline_plots(y_coords, vertical_u, x_coords, horizontal_v)

        return vertical_u, horizontal_v

    def _export_centerline_data(self, y_coords, vertical_u, vertical_v,
                                x_coords, horizontal_u, horizontal_v):
        """导出中心线速度分布数据到表格文件"""
        print("📊 导出中心线速度分布数据...")

        # 创建垂直中心线数据表
        vertical_data = {
            'Y_Coordinate': y_coords,
            'U_Velocity_m/s': vertical_u,
            'V_Velocity_m/s': vertical_v,
            'Velocity_Magnitude_m/s': np.sqrt(vertical_u ** 2 + vertical_v ** 2),
            'Y_Normalized': y_coords / (len(y_coords) - 1)  # 归一化坐标
        }
        vertical_df = pd.DataFrame(vertical_data)

        # 创建水平中心线数据表
        horizontal_data = {
            'X_Coordinate': x_coords,
            'U_Velocity_m/s': horizontal_u,
            'V_Velocity_m/s': horizontal_v,
            'Velocity_Magnitude_m/s': np.sqrt(horizontal_u ** 2 + horizontal_v ** 2),
            'X_Normalized': x_coords / (len(x_coords) - 1)  # 归一化坐标
        }
        horizontal_df = pd.DataFrame(horizontal_data)

        # 保存为CSV文件
        vertical_csv_path = os.path.join(self.analysis_dir, 'vertical_centerline_velocity_data.csv')
        horizontal_csv_path = os.path.join(self.analysis_dir, 'horizontal_centerline_velocity_data.csv')

        vertical_df.to_csv(vertical_csv_path, index=False, float_format='%.8f')
        horizontal_df.to_csv(horizontal_csv_path, index=False, float_format='%.8f')

        # 保存为Excel文件（包含两个工作表）
        if HAS_OPENPYXL:
            try:
                excel_path = os.path.join(self.analysis_dir, 'centerline_velocity_data.xlsx')
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    vertical_df.to_excel(writer, sheet_name='Vertical_Centerline', index=False)
                    horizontal_df.to_excel(writer, sheet_name='Horizontal_Centerline', index=False)
                print(f"✅ Excel文件已保存: {excel_path}")
            except Exception as e:
                print(f"⚠️ Excel导出失败: {e}")
        else:
            print("⚠️ 跳过Excel导出 (需要安装openpyxl: pip install openpyxl)")


        # 创建统计摘要
        summary_data = {
            'Profile_Type': ['Vertical_Centerline_U', 'Vertical_Centerline_V',
                             'Horizontal_Centerline_U', 'Horizontal_Centerline_V'],
            'Max_Value_m/s': [np.max(vertical_u), np.max(vertical_v),
                              np.max(horizontal_u), np.max(horizontal_v)],
            'Min_Value_m/s': [np.min(vertical_u), np.min(vertical_v),
                              np.min(horizontal_u), np.min(horizontal_v)],
            'Mean_Value_m/s': [np.mean(vertical_u), np.mean(vertical_v),
                               np.mean(horizontal_u), np.mean(horizontal_v)],
            'Std_Dev_m/s': [np.std(vertical_u), np.std(vertical_v),
                            np.std(horizontal_u), np.std(horizontal_v)]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(self.analysis_dir, 'centerline_velocity_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False, float_format='%.8f')

        print(f"✅ 垂直中心线数据已保存: {vertical_csv_path}")
        print(f"✅ 水平中心线数据已保存: {horizontal_csv_path}")
        print(f"✅ 统计摘要已保存: {summary_csv_path}")

    def _create_streamline_plot(self, final_vel, z_mid):
        """创建XY平面流线图（不显示横纵坐标）"""
        print("🌊 创建XY平面流线图...")

        Nx, Ny, Nz = final_vel.shape[:3]

        # 提取XY平面的速度分量
        xy_u = final_vel[:, :, z_mid, 0]
        xy_v = final_vel[:, :, z_mid, 1]

        # 计算速度大小
        velocity_magnitude = np.sqrt(xy_u ** 2 + xy_v ** 2)

        # 创建坐标网格
        x_coords = np.arange(Nx)
        y_coords = np.arange(Ny)
        X, Y = np.meshgrid(x_coords, y_coords)

        # 创建流线图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # 第一个子图：带速度大小背景的流线图
        contour = ax1.contourf(X, Y, velocity_magnitude.T, levels=20, cmap=self.cfd_cmap, alpha=0.8)
        streamplot1 = ax1.streamplot(X, Y, xy_u.T, xy_v.T,
                                     density=2.0, color='white', linewidth=1.2,
                                     arrowsize=1.5, arrowstyle='->')

        ax1.set_title(f'XY Plane Streamlines with Velocity Magnitude\n(Z={z_mid}, Re={self.Re})',
                      fontsize=14, fontweight='bold')
        # 移除横纵坐标标签
        # ax1.set_xlabel('X Coordinate', fontsize=12)  # 注释或删除X轴标签
        # ax1.set_ylabel('Y Coordinate', fontsize=12)  # 注释或删除Y轴标签
        ax1.set_aspect('equal')

        # 隐藏刻度线和刻度标签
        ax1.set_xticks([])  # 移除X轴刻度
        ax1.set_yticks([])  # 移除Y轴刻度
        ax1.spines['top'].set_visible(False)  # 隐藏上边框
        ax1.spines['right'].set_visible(False)  # 隐藏右边框
        ax1.spines['bottom'].set_visible(False)  # 隐藏下边框
        ax1.spines['left'].set_visible(False)  # 隐藏左边框

        # 添加颜色条
        cbar1 = plt.colorbar(contour, ax=ax1, shrink=0.8)
        cbar1.set_label('Velocity Magnitude [m/s]', fontsize=11)

        # 第二个子图：纯流线图（按速度大小着色）
        speed = np.sqrt(xy_u ** 2 + xy_v ** 2)
        streamplot2 = ax2.streamplot(X, Y, xy_u.T, xy_v.T,
                                     color=speed.T, density=2.5, cmap=self.cfd_cmap,
                                     linewidth=1.5, arrowsize=1.5, arrowstyle='->')

        ax2.set_title(f'XY Plane Streamlines (Colored by Speed)\n(Z={z_mid}, Re={self.Re})',
                      fontsize=14, fontweight='bold')
        # 移除横纵坐标标签
        # ax2.set_xlabel('X Coordinate', fontsize=12)  # 注释或删除X轴标签
        # ax2.set_ylabel('Y Coordinate', fontsize=12)  # 注释或删除Y轴标签
        ax2.set_aspect('equal')

        # 隐藏刻度线和刻度标签
        ax2.set_xticks([])  # 移除X轴刻度
        ax2.set_yticks([])  # 移除Y轴刻度
        ax2.spines['top'].set_visible(False)  # 隐藏上边框
        ax2.spines['right'].set_visible(False)  # 隐藏右边框
        ax2.spines['bottom'].set_visible(False)  # 隐藏下边框
        ax2.spines['left'].set_visible(False)  # 隐藏左边框

        # 添加颜色条
        cbar2 = plt.colorbar(streamplot2.lines, ax=ax2, shrink=0.8)
        cbar2.set_label('Velocity  [m/s]', fontsize=11)

        # 添加边界框以显示空腔边界（保持不变）
        for ax in [ax1, ax2]:
            ax.plot([0, Nx - 1, Nx - 1, 0, 0], [0, 0, Ny - 1, Ny - 1, 0],
                    'k-', linewidth=2, alpha=0.7, label='Cavity Boundary')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)  # 如需移除网格可注释此行

        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(self.analysis_dir, 'xy_streamline_plot.png'),
                    dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()

        # 创建高分辨率流线图（保持不变）
        self._create_high_resolution_streamline(xy_u, xy_v, velocity_magnitude)

        print("✅ 流线图已保存")

    def _create_high_resolution_streamline(self, xy_u, xy_v, velocity_magnitude):
        """创建高分辨率单一流线图"""
        if not HAS_SCIPY:
            print("⚠️ 跳过高分辨率流线图生成 (需要安装scipy: pip install scipy)")
            return

        try:
            Nx, Ny = xy_u.shape

            # 创建更细密的网格用于插值
            x_fine = np.linspace(0, Nx - 1, Nx * 2)
            y_fine = np.linspace(0, Ny - 1, Ny * 2)
            X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

            # 插值速度场到更细的网格
            x_orig = np.arange(Nx)
            y_orig = np.arange(Ny)
            X_orig, Y_orig = np.meshgrid(x_orig, y_orig)

            points_orig = np.column_stack((X_orig.ravel(), Y_orig.ravel()))
            u_fine = griddata(points_orig, xy_u.T.ravel(), (X_fine, Y_fine), method='cubic')
            v_fine = griddata(points_orig, xy_v.T.ravel(), (X_fine, Y_fine), method='cubic')
            vel_mag_fine = griddata(points_orig, velocity_magnitude.T.ravel(), (X_fine, Y_fine), method='cubic')

            # 创建高质量流线图
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))

            # 背景速度大小等高线
            contour = ax.contourf(X_fine, Y_fine, vel_mag_fine, levels=30,
                                  cmap=self.cfd_cmap, alpha=0.9)

            # 流线
            streamplot = ax.streamplot(X_fine, Y_fine, u_fine, v_fine,
                                       density=3.0, color='white', linewidth=1.0,
                                       arrowsize=1.2, arrowstyle='->')

            # 设置图形属性（移除坐标相关元素）
            ax.set_title(f'High-Resolution XY Streamlines\n(Re={self.Re})',
                         fontsize=16, fontweight='bold', pad=20)
            # 移除坐标标签
            # ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')  # 注释X轴标签
            # ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')  # 注释Y轴标签
            ax.set_aspect('equal')

            # 隐藏刻度线和刻度标签
            ax.set_xticks([])  # 移除X轴刻度
            ax.set_yticks([])  # 移除Y轴刻度

            # 隐藏所有边框
            for spine in ax.spines.values():
                spine.set_visible(False)  # 不显示坐标轴边框

            # 颜色条
            cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Velocity Magnitude [m/s]', fontsize=12, fontweight='bold')

            # 添加边界框
            Nx_orig, Ny_orig = xy_u.shape
            ax.plot([0, Nx_orig - 1, Nx_orig - 1, 0, 0], [0, 0, Ny_orig - 1, Ny_orig - 1, 0],
                    'k-', linewidth=3, alpha=0.8, label='Cavity Boundary')

            # 可选：移除网格（如需保留可注释此行）
            # ax.grid(True, alpha=0.2, linestyle='--')

            ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

            # 移除刻度样式设置（已无刻度，无需设置）
            # ax.tick_params(axis='both', which='major', labelsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(self.analysis_dir, 'high_resolution_streamline.png'),
                        dpi=600, bbox_inches='tight', facecolor='white')
            plt.close()

            print("✅ 高分辨率流线图已保存")

        except Exception as e:
            print(f"⚠️ 高分辨率流线图生成失败: {e}")

    def _create_centerline_plots(self, y_coords, vertical_u, x_coords, horizontal_v):
        """创建原有的中心线分析图"""
        # 创建专业CFD风格的图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('white')

        # 垂直中心线U速度分布
        axes[0].plot(vertical_u, y_coords, 'b-', linewidth=2.5, label='LBM Results', alpha=0.8)
        axes[0].fill_betweenx(y_coords, 0, vertical_u, alpha=0.2, color='blue')
        axes[0].set_xlabel('U Velocity [m/s]', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Vertical Centerline U-velocity Profile\n(Re = {self.Re})',
                          fontsize=13, fontweight='bold', pad=20)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(fontsize=11)

        # 添加统计信息
        u_max = np.max(vertical_u)
        u_min = np.min(vertical_u)
        axes[0].text(0.02, 0.98, f'Max U: {u_max:.6f} m/s\nMin U: {u_min:.6f} m/s',
                     transform=axes[0].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                     fontsize=10)

        # 水平中心线V速度分布
        axes[1].plot(x_coords, horizontal_v, 'r-', linewidth=2.5, label='LBM Results', alpha=0.8)
        axes[1].fill_between(x_coords, 0, horizontal_v, alpha=0.2, color='red')
        axes[1].set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('V Velocity [m/s]', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Horizontal Centerline V-velocity Profile\n(Re = {self.Re})',
                          fontsize=13, fontweight='bold', pad=20)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(fontsize=11)

        # 添加统计信息
        v_max = np.max(horizontal_v)
        v_min = np.min(horizontal_v)
        axes[1].text(0.02, 0.02, f'Max V: {v_max:.6f} m/s\nMin V: {v_min:.6f} m/s',
                     transform=axes[1].transAxes, verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
                     fontsize=10)

        # 设置坐标轴样式
        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)

        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(self.analysis_dir, 'centerline_profiles.png'),
                    dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_additional_streamline_analysis(self):
        """创建额外的流线分析图（无坐标轴显示）"""
        print("🔄 创建额外流线分析...")

        final_vel = self.velocity_data[-1]
        Nx, Ny, Nz = final_vel.shape[:3]
        z_mid = Nz // 2

        # 提取XY平面速度
        xy_u = final_vel[:, :, z_mid, 0]
        xy_v = final_vel[:, :, z_mid, 1]
        velocity_magnitude = np.sqrt(xy_u ** 2 + xy_v ** 2)

        # 创建多种流线图
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        x_coords = np.arange(Nx)
        y_coords = np.arange(Ny)
        X, Y = np.meshgrid(x_coords, y_coords)

        # 1. 基础流线图
        axes[0, 0].streamplot(X, Y, xy_u.T, xy_v.T, density=2.0, color='blue',
                              linewidth=1.0, arrowsize=1.2)
        axes[0, 0].set_title('Basic Streamlines', fontsize=12, fontweight='bold')
        # 移除坐标标签
        # axes[0, 0].set_xlabel('X')
        # axes[0, 0].set_ylabel('Y')
        axes[0, 0].set_aspect('equal')
        axes[0, 0].grid(True, alpha=0.3)

        # 隐藏刻度和边框
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        for spine in axes[0, 0].spines.values():
            spine.set_visible(False)

        # 2. 按速度着色的流线图
        speed = np.sqrt(xy_u ** 2 + xy_v ** 2)
        strm = axes[0, 1].streamplot(X, Y, xy_u.T, xy_v.T, color=speed.T,
                                     density=2.0, cmap='viridis', linewidth=1.5)
        axes[0, 1].set_title('Streamlines Colored by Speed', fontsize=12, fontweight='bold')
        # 移除坐标标签
        # axes[0, 1].set_xlabel('X')
        # axes[0, 1].set_ylabel('Y')
        axes[0, 1].set_aspect('equal')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(strm.lines, ax=axes[0, 1], label='Speed [m/s]')

        # 隐藏刻度和边框
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        for spine in axes[0, 1].spines.values():
            spine.set_visible(False)

        # 3. 涡度等高线 + 流线
        # 计算涡度
        vorticity = np.zeros((Nx - 2, Ny - 2))
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                dvdx = (xy_v[i + 1, j] - xy_v[i - 1, j]) / 2.0
                dudy = (xy_u[i, j + 1] - xy_u[i, j - 1]) / 2.0
                vorticity[i - 1, j - 1] = dvdx - dudy

        # 创建涡度网格
        x_vort = np.arange(1, Nx - 1)
        y_vort = np.arange(1, Ny - 1)
        X_vort, Y_vort = np.meshgrid(x_vort, y_vort)

        contour = axes[1, 0].contourf(X_vort, Y_vort, vorticity.T, levels=20,
                                      cmap='RdBu_r', alpha=0.7)
        axes[1, 0].streamplot(X, Y, xy_u.T, xy_v.T, density=1.5, color='black',
                              linewidth=0.8, arrowsize=1.0)
        axes[1, 0].set_title('Vorticity + Streamlines', fontsize=12, fontweight='bold')
        # 移除坐标标签
        # axes[1, 0].set_xlabel('X')
        # axes[1, 0].set_ylabel('Y')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(contour, ax=axes[1, 0], label='Vorticity [1/s]')

        # 隐藏刻度和边框
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        for spine in axes[1, 0].spines.values():
            spine.set_visible(False)

        # 4. 速度大小等高线 + 流线
        contour2 = axes[1, 1].contourf(X, Y, velocity_magnitude.T, levels=20,
                                       cmap=self.cfd_cmap, alpha=0.8)
        axes[1, 1].streamplot(X, Y, xy_u.T, xy_v.T, density=2.0, color='white',
                              linewidth=1.0, arrowsize=1.2)
        axes[1, 1].set_title('Velocity Magnitude + Streamlines', fontsize=12, fontweight='bold')
        # 移除坐标标签
        # axes[1, 1].set_xlabel('X')
        # axes[1, 1].set_ylabel('Y')
        axes[1, 1].set_aspect('equal')
        plt.colorbar(contour2, ax=axes[1, 1], label='Velocity [m/s]')

        # 隐藏刻度和边框
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        for spine in axes[1, 1].spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'comprehensive_streamline_analysis.png'),
                    dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()

    def export_flow_field_data(self):
        """导出完整的流场数据"""
        print("📤 导出完整流场数据...")

        final_vel = self.velocity_data[-1]
        final_p = self.pressure_data[-1]
        final_rho = self.density_data[-1]

        Nx, Ny, Nz = final_vel.shape[:3]
        z_mid = Nz // 2

        # 创建坐标网格
        x_coords = np.arange(Nx)
        y_coords = np.arange(Ny)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

        # 提取XY中心切片的数据
        xy_data = {
            'X_Coordinate': X.ravel(),
            'Y_Coordinate': Y.ravel(),
            'U_Velocity_m/s': final_vel[:, :, z_mid, 0].ravel(),
            'V_Velocity_m/s': final_vel[:, :, z_mid, 1].ravel(),
            'W_Velocity_m/s': final_vel[:, :, z_mid, 2].ravel(),
            'Pressure_Pa': final_p[:, :, z_mid].ravel(),
            'Density_kg/m3': final_rho[:, :, z_mid].ravel()
        }

        # 计算衍生量
        velocity_magnitude = np.sqrt(final_vel[:, :, z_mid, 0] ** 2 +
                                     final_vel[:, :, z_mid, 1] ** 2 +
                                     final_vel[:, :, z_mid, 2] ** 2)
        xy_data['Velocity_Magnitude_m/s'] = velocity_magnitude.ravel()

        # 创建DataFrame并保存
        flow_df = pd.DataFrame(xy_data)
        csv_path = os.path.join(self.analysis_dir, f'xy_plane_flow_field_data_z{z_mid}.csv')
        flow_df.to_csv(csv_path, index=False, float_format='%.8f')

        print(f"✅ XY平面流场数据已保存: {csv_path}")

        # 保存为压缩格式（对于大数据集）
        try:
            parquet_path = os.path.join(self.analysis_dir, f'xy_plane_flow_field_data_z{z_mid}.parquet')
            flow_df.to_parquet(parquet_path, index=False)
            print(f"✅ 压缩格式数据已保存: {parquet_path}")
        except Exception as e:
            print(f"⚠️ Parquet格式保存失败: {e}, 仅保存CSV格式")


def main():
    """主函数 - 运行完整的后处理分析"""

    # 设置参数
    Re = 400
    data_dir = f'../lid_driven_cavity_flow/Re{Re}/'

    print("🔬 3D顶盖驱动空腔流动 - 后处理分析")
    print("=" * 60)

    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请先运行主仿真程序生成数据!")
        return

    # 创建分析器
    analyzer = CavityFlowAnalyzer(data_dir, Re)

    try:
        # 加载数据
        analyzer.load_simulation_data()

        if len(analyzer.time_series) == 0:
            print("❌ 未找到仿真数据文件!")
            return
        analyzer.analyze_convergence()
        analyzer.analyze_flow_features()
        analyzer.create_slice_visualizations()
        analyzer.create_3d_visualization()
        analyzer.analyze_center_line_profiles()


        analyzer.create_additional_streamline_analysis()
        analyzer.export_flow_field_data()



        print("\n🎉 后处理分析完成!")
        print(f"📊 分析报告: {analyzer.analysis_dir}/analysis_report.html")
        print(f"📈 可视化图像保存在: {analyzer.analysis_dir}/")
        print("\n📋 新增功能总结:")
        print("  ✅ 中心线速度分布数据已导出为CSV和其他格式文件")
        print("  ✅ XY平面流线图已生成")
        print("  ✅ 综合流线分析图已创建")
        print("  ✅ 完整流场数据已导出")


    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()


# 使用示例和主程序
if __name__ == "__main__":
    main()