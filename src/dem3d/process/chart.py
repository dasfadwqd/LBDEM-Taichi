'''
Module responsible for generating images through data processing
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import json

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'


def read_p4p_file(file_name: str, particle_ids: list):
    """
    解析.p4p文件，提取指定粒子的时序运动数据

    Args:
        file_name: P4P文件路径
        particle_ids: 需要提取的粒子ID列表

    Returns:
        dict: 包含各粒子时序数据的字典
    """
    # 初始化数据结构
    particle_data = {pid: {"time": [], "vx": [], "vy": [], "vz": [],
                           "px": [], "py": [], "pz": []}
                     for pid in particle_ids}

    # 使用集合加速ID查找
    particle_ids_set = set(particle_ids)

    current_time = None
    total_lines = 0
    data_lines = 0
    skipped_lines = 0

    print(f"开始解析文件: {file_name}")
    print(f"目标粒子ID: {particle_ids}")

    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                total_lines += 1
                line = line.strip()

                # 跳过空行
                if not line:
                    continue

                # 检测时间步标记
                if line.startswith("TIMESTEP"):
                    try:
                        # 读取下一行的时间信息
                        time_line = next(file).strip()
                        total_lines += 1
                        time_info = time_line.split()
                        current_time = float(time_info[0])
                        print(f"找到时间步: {current_time}")
                    except (StopIteration, ValueError, IndexError) as e:
                        print(f"警告: 第{line_num + 1}行时间信息解析失败: {e}")
                        continue

                # 跳过表头
                elif line.startswith("ID") and "GROUP" in line and "RAD" in line:
                    continue

                # 解析数据行
                else:
                    try:
                        parts = line.split()

                        # 检查数据完整性
                        if len(parts) != 10:
                            skipped_lines += 1
                            if len(parts) > 0:  # 不是空行
                                print(f"警告: 第{line_num}行数据字段不完整 ({len(parts)}/10): {line}")
                            continue

                        # 解析各字段
                        ID = int(float(parts[0]))  # 先转float再转int，处理"1.0"格式
                        group, rad, mass, px, py, pz, vx, vy, vz = map(float, parts[1:])

                        # 检查是否是目标粒子
                        if ID in particle_ids_set:
                            if current_time is not None:
                                particle_data[ID]["time"].append(current_time)
                                particle_data[ID]["vx"].append(vx)
                                particle_data[ID]["vy"].append(vy)
                                particle_data[ID]["vz"].append(vz)
                                particle_data[ID]["px"].append(px)
                                particle_data[ID]["py"].append(py)
                                particle_data[ID]["pz"].append(pz)

                                data_lines += 1
                            else:
                                print(f"警告: 第{line_num}行找到粒子数据但没有时间信息")

                    except ValueError as e:
                        skipped_lines += 1
                        print(f"警告: 第{line_num}行数值解析失败: {line}")
                        print(f"错误详情: {e}")
                    except IndexError as e:
                        skipped_lines += 1
                        print(f"警告: 第{line_num}行数据字段不足: {line}")

    except FileNotFoundError:
        print(f"错误: 文件未找到 - {file_name}")
        return {}
    except Exception as e:
        print(f"错误: 读取文件时发生异常 - {e}")
        return {}

    # 统计信息
    print(f"\n=== 解析完成 ===")
    print(f"总行数: {total_lines}")
    print(f"有效数据行数: {data_lines}")
    print(f"跳过行数: {skipped_lines}")

    return particle_data

# 解析output.p4c文件，提取接触力信息
def read_p4c_file(file_name: str):
    contact_data = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("TIMESTEP  CONTACTS"):
                time_info = next(file).strip().split()
                current_time = float(time_info[0])
            elif line.startswith("P1  P2  CX  CY  CZ  FX  FY  FZ "):
                continue
            else:
                try:
                    p1, p2, cx, cy, cz, fx, fy ,fz = map(float, line.split())
                    contact_data.append((current_time, p1, p2,cx ,cy ,cz, fx, fy,fz))
                except ValueError:
                    print(f"Warning: Skipping invalid line - {line}")
    return contact_data

# 绘制速度与时间的关系图
def plot_speed_vs_time(particle_data, save_path):
    plt.figure(figsize=(10, 6))  # 增大图表尺寸

    colors = plt.cm.tab10(np.linspace(0, 1, len(particle_data)))

    for (pid, data), color in zip(particle_data.items(), colors):
        times = data["time"]
        vxs = data["vx"]
        vys = data["vy"]
        vzs = data["vz"]
        plt.plot(times, vxs, label=f"Particle {pid} (VX)", linewidth=1.5, alpha=0.9, color=color, linestyle='-')
        plt.plot(times, vys, label=f"Particle {pid} (VY)", linewidth=1.5, alpha=0.9, color=color, linestyle='-.')
        plt.plot(times, vzs, label=f"Particle {pid} (VZ)", linewidth=1.5, alpha=0.9, color=color, linestyle='--')

    plt.xlabel("t (s)", fontsize=14)  # 添加单位
    plt.ylabel("$V$ (m/s)", fontsize=14)  # 添加单位

    plt.title("V/t", fontsize=16)  # 添加标题

    # 设置次刻度
    ax = plt.gca()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # 设置刻度标签格式
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    # 设置刻度线样式
    ax.tick_params(axis='both', which='major', direction='in', labelsize=12, width=1.2, length=6)
    ax.tick_params(axis='both', which='minor', direction='in', width=1, length=3)

    # 设置网格样式
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.grid(True, which='minor', linestyle='--', alpha=0.3)
    plt.xlim(left=0)
    # 调整图例位置及透明度
    plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.15), fontsize=14, framealpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "speed_vs_time.png"), dpi=300, bbox_inches='tight')
    plt.show()
# 位移随时间
def plot_displacement_vs_time(particle_data, save_path):
    plt.figure(figsize=(8, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(particle_data)))

    for (pid, data), color in zip(particle_data.items(), colors):
        times = data["time"]
        pxs = data["px"]
        pys = data["py"]
        pzs = data["pz"]



        plt.plot(times, pxs, label=f"Particle {pid} (X)", linewidth=2, alpha=0.8, color=color,
                 linestyle='-')
        plt.plot(times, pys, label=f"Particle {pid} (Y)", linewidth=2, alpha=0.8, color=color,
                 linestyle='-.')
        plt.plot(times, pzs, label=f"Particle {pid} (Y)", linewidth=2, alpha=0.8, color=color,
                 linestyle='--')

    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Displacement", fontsize=14)
    plt.title("Particle Displacement vs Time", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "displacement_vs_time.png"), dpi=300)
    plt.show()

# 绘制接触力与位移的关系图
def plot_force_vs_displacement(particle_data, contact_data, save_path):
    plt.figure(figsize=(8, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(particle_data)))

    for pid, color in zip(particle_data.keys(), colors):
        times = particle_data[pid]["time"]
        pxs = particle_data[pid]["px"]
        pys = particle_data[pid]["py"]
        initial_px = pxs[0]
        initial_py = pys[0]
        displacements = [np.sqrt((py - initial_py)**2 + 0) *1e6 for px, py in zip(pxs, pys)]

        # 初始化forces列表，长度与displacements一致，初始值为0
        forces = [0.0] * len(displacements)

        # 遍历接触力数据，填充forces列表
        for time, p1, p2, fx, fy in contact_data:
            if p1 == pid or p2 == pid:
                force = np.sqrt(0 + fy**2)/1000
                # 找到对应的时间步索引
                if time in times:
                    index = times.index(time)
                    forces[index] = force

        # 确保displacements和forces长度一致
        assert len(displacements) == len(forces), f"Displacement and force data length mismatch for particle {pid}"

        plt.plot(displacements, forces, label=f"Particle {pid}", linewidth=2, alpha=0.8, color=color)

    max_displacement = max(displacements)  # 获取最大位移值
    x_values = np.linspace(0, max_displacement, 100)  # 生成100个点
    y_values = 654.72 * x_values   # 计算对应的y值

    #plt.plot(x_values, y_values, color='#98CA84', linestyle='--',linewidth=2, label='Kt*x')
    plt.xlabel("Displacement", fontsize=14)
    plt.ylabel("Force", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "force_vs_displacement.png"), dpi=300)
    plt.show()

# 绘制流体与时间的关系图
def plot_fluid_force_vs_time(particle_data, save_path):
    plt.figure(figsize=(10, 6))  # 增大图表尺寸

    colors = plt.cm.tab10(np.linspace(0, 1, len(particle_data)))  # 使用更专业的颜色映射

    for (pid, data), color in zip(particle_data.items(), colors):
        times = data["time"]
        fx = data["Fx"]
        fy = data["Fy"]
        plt.plot(times, fx, label=f"Particle {pid} (VX)", linewidth=1.5, alpha=0.9, color="red", linestyle='-')
        plt.plot(times, fy, label=f"Particle {pid} (VY)", linewidth=1.5, alpha=0.9, color=color, linestyle='-')

    plt.xlabel("t (s)", fontsize=14)  # 添加单位
    plt.ylabel("$F_y$ (m/s)", fontsize=14)  # 添加单位

    plt.title("Force_Fliud", fontsize=16)  # 添加标题

    # 设置次刻度
    ax = plt.gca()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # 设置刻度标签格式
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    # 设置刻度线样式
    ax.tick_params(axis='both', which='major', direction='in', labelsize=12, width=1.2, length=6)
    ax.tick_params(axis='both', which='minor', direction='in', width=1, length=3)

    # 设置网格样式
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.grid(True, which='minor', linestyle='--', alpha=0.3)
    plt.xlim(left=0)
    # 调整图例位置及透明度
    plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.15), fontsize=14, framealpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "force_fluid.png"), dpi=300, bbox_inches='tight')
    plt.show()
# 绘制接触力与时间的关系图
def plot_force_vs_time(particle_data, contact_data, save_path):
    plt.figure(figsize=(8, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(particle_data)))

    # 为每个粒子准备力-时间数据
    force_time_data = {pid: {"time": [], "force": []} for pid in particle_data.keys()}

    # 从接触力数据中提取每个粒子在每个时间步的合力
    for time, p1, p2, fx, fy in contact_data:
        # 计算合力大小
        force_magnitude = np.sqrt(0 ** 2 + fy ** 2) / 1000

        # 将力添加到相应粒子的数据中
        for pid in [p1, p2]:
            if pid in force_time_data:
                # 检查这个时间步是否已经存在
                if time in force_time_data[pid]["time"]:
                    idx = force_time_data[pid]["time"].index(time)
                    # 累加同一时间步的力
                    force_time_data[pid]["force"][idx] += force_magnitude
                else:
                    force_time_data[pid]["time"].append(time)
                    force_time_data[pid]["force"].append(force_magnitude)

    # 将绘图循环移到数据收集循环外部
    # 绘制每个粒子的力-时间曲线
    for (pid, data), color in zip(force_time_data.items(), colors):
        # 确保时间序列有序
        time_force_pairs = sorted(zip(data["time"], data["force"]))
        if time_force_pairs:  # 确保有数据再绘图
            times, forces = zip(*time_force_pairs)
            # 将时间转换为微秒
            times_microsec = [t * 1e6 for t in times]  # 假设原始时间单位为秒
            if times_microsec[0] != 0:
                times_microsec = [0] + list(times_microsec)
                forces = [0] + list(forces)
            plt.plot(times_microsec, forces, label=f"Particle {pid}", linewidth=2.0, alpha=0.9, color=color)

    plt.xlabel("Time(us)", fontsize=16)
    plt.ylabel("Normal contact Force (kN)", fontsize=16)
    plt.title("Glass", fontsize=16, fontweight='bold')
    plt.legend(fontsize=14, frameon=True, facecolor='white', edgecolor='black')  # 增强图例可见性
    plt.grid(True, linestyle="-", alpha=0.3)  # 调整网格线

    # 设置刻度字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # 设置y轴起点为0
    plt.ylim(bottom=0)  # 这行代码确保y轴从0开始
    plt.xlim(left=0)

    # 添加边框
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)

    plt.gca().set_facecolor('#f8f9fa')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "force_vs_time.png"), dpi=400)
    plt.show()



# 新增：提取指定粒子的指定数据
def extract_specific_data(particle_data, particle_ids=None, data_types=None):
    """
    提取指定粒子的指定类型数据

    参数:
    particle_data - 通过read_p4p_file读取的粒子数据字典
    particle_ids - 要提取的粒子ID列表，如果为None则提取所有粒子
    data_types - 要提取的数据类型列表，如'time', 'vx', 'vy','vz', 'px', 'py', 'pz'，如果为None则提取所有类型

    返回:
    包含指定数据的字典
    """
    result = {}

    # 如果未指定粒子ID，则使用所有可用的粒子ID
    if particle_ids is None:
        particle_ids = list(particle_data.keys())

    # 确保particle_ids是列表
    if not isinstance(particle_ids, list):
        particle_ids = [particle_ids]

    # 如果未指定数据类型，则使用所有可用的数据类型
    if data_types is None:
        # 假设所有粒子有相同的数据类型
        if particle_data and len(particle_data) > 0:
            first_pid = list(particle_data.keys())[0]
            data_types = list(particle_data[first_pid].keys())

    # 确保data_types是列表
    if not isinstance(data_types, list):
        data_types = [data_types]

    # 提取数据
    for pid in particle_ids:
        if pid in particle_data:
            result[pid] = {}
            for data_type in data_types:
                if data_type in particle_data[pid]:
                    result[pid][data_type] = particle_data[pid][data_type]

    return result


def plot_physical_distance_vs_time(particle_data, save_path, particle_diameter=None):
    """
    绘制两颗粒间物理距离 dr_phy,s 与时间的关系图

    参数:
    particle_data - 通过read_p4p_file读取的粒子数据字典
    save_path - 保存路径
    particle_diameter - 颗粒直径 Dp，如果为None则需要用户输入
    """
    # 检查是否有两个颗粒
    particle_ids = list(particle_data.keys())
    if len(particle_ids) != 2:
        print(f"错误: 需要两个颗粒进行距离计算，但找到了{len(particle_ids)}个颗粒")
        return

    pid1, pid2 = particle_ids
    data1 = particle_data[pid1]
    data2 = particle_data[pid2]

    # 检查时间长度是否一致
    if len(data1["time"]) != len(data2["time"]):
        print("错误: 两个颗粒的时间序列长度不一致")
        return

    # 如果未提供颗粒直径，询问用户输入
    if particle_diameter is None:
        try:
            particle_diameter = float(input("请输入颗粒直径 Dp (m): "))
        except ValueError:
            print("输入的直径值无效，使用默认值 0.001 m")
            particle_diameter = 0.001

    # 计算两颗粒间的距离
    times = data1["time"]
    dx_values = []
    dy_values = []
    dz_values = []
    dr_phy_s_values = []

    for i in range(len(times)):
        # 计算两颗粒在各方向上的距离
        dx = data1["px"][i] - data2["px"][i]
        dy = data1["py"][i] - data2["py"][i]
        dz = data1["pz"][i] - data2["pz"][i]

        dx_values.append(dx)
        dy_values.append(dy)
        dz_values.append(dz)

        # 计算物理距离 dr_phy,s = sqrt(dx^2 + dy^2 + dz^2) - Dp
        distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        dr_phy_s = distance - particle_diameter
        dr_phy_s_values.append(dr_phy_s)

    # 绘制图形
    plt.figure(figsize=(10, 6))

    plt.plot(times, dr_phy_s_values, label=f'$dr_{{phy,s}}$ (Particles {pid1}-{pid2})',
             linewidth=2, alpha=0.9, color='red', linestyle='-')

    plt.xlabel("t (s)", fontsize=14)
    plt.ylabel("$dr_{phy,s}$ (m)", fontsize=14)
    plt.title(f"Physical Distance vs Time\n($D_p$ = {particle_diameter:.6f} m)", fontsize=16)

    # 设置次刻度
    ax = plt.gca()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # 设置刻度标签格式
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    # 设置刻度线样式
    ax.tick_params(axis='both', which='major', direction='in', labelsize=12, width=1.2, length=6)
    ax.tick_params(axis='both', which='minor', direction='in', width=1, length=3)

    # 设置网格样式
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.grid(True, which='minor', linestyle='--', alpha=0.3)
    plt.xlim(left=0)

    # 添加图例
    plt.legend(loc='best', fontsize=12, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "physical_distance_vs_time.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # 输出一些统计信息
    print(f"\n=== 物理距离统计信息 ===")
    print(f"颗粒对: {pid1} - {pid2}")
    print(f"颗粒直径 Dp: {particle_diameter:.6f} m")
    print(f"最小物理距离: {min(dr_phy_s_values):.6f} m")
    print(f"最大物理距离: {max(dr_phy_s_values):.6f} m")
    print(f"平均物理距离: {np.mean(dr_phy_s_values):.6f} m")

    return {
        'times': times,
        'dx': dx_values,
        'dy': dy_values,
        'dz': dz_values,
        'dr_phy_s': dr_phy_s_values,
        'particle_diameter': particle_diameter
    }

# 导出指定数据到CSV
def export_specific_data_to_csv(particle_data, output_path, particle_ids=None, data_types=None):
    """
    将指定粒子的指定类型数据导出为单个CSV文件

    参数:
    particle_data - 通过read_p4p_file读取的粒子数据字典
    output_path - 输出文件路径
    particle_ids - 要导出的粒子ID列表，如果为None则导出所有粒子
    data_types - 要导出的数据类型列表，如果为None则导出所有类型
    """
    # 提取指定数据
    extracted_data = extract_specific_data(particle_data, particle_ids, data_types)

    # 准备数据帧
    df_dict = {}

    # 所有粒子共享相同的时间轴
    if extracted_data and 'time' in next(iter(extracted_data.values())):
        df_dict['time'] = next(iter(extracted_data.values()))['time']

    # 为每个粒子的每种数据类型创建列
    for pid, data in extracted_data.items():
        for data_type, values in data.items():
            if data_type != 'time':  # 时间已经添加
                column_name = f"{data_type}_particle_{pid}"
                df_dict[column_name] = values

    # 创建数据帧并导出
    df = pd.DataFrame(df_dict)
    df.to_csv(output_path, index=False)
    print(f"已将指定数据导出到 {output_path}")


def get_all_particle_ids(file_name: str):
    """
    获取P4P文件中所有的粒子ID
    Args:
        file_name: P4P文件路径
    Returns:
        list: 所有粒子ID的列表
    """
    particle_ids = set()

    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()

                # 跳过空行和时间步标记
                if not line or line.startswith("TIMESTEP"):
                    continue

                # 跳过表头
                if line.startswith("ID") and "GROUP" in line and "RAD" in line:
                    continue

                # 解析数据行
                try:
                    parts = line.split()
                    if len(parts) == 10:  # 确保是完整的数据行
                        particle_id = int(float(parts[0]))
                        particle_ids.add(particle_id)
                except (ValueError, IndexError):
                    continue

    except FileNotFoundError:
        print(f"错误: 文件未找到 - {file_name}")
        return []
    except Exception as e:
        print(f"错误: 读取文件时发生异常 - {e}")
        return []

    return sorted(list(particle_ids))


# 主函数
def main():
    # 输入文件名
    p4p_file_name = r"D:\CHen\LBDEM-Taichi\example\lbdem\dkt\output.p4p"
    p4c_file_name = r"D:\CHen\LBDEM-Taichi\example\unlbdem\single_particle_settle\Re1.5_un\output.p4c"

    # 获取文件中所有的粒子ID
    all_particle_ids = get_all_particle_ids(p4p_file_name)

    if not all_particle_ids:
        print("未找到任何粒子ID，请检查文件路径和格式。")
        return

    print(f"文件中发现的所有粒子ID: {all_particle_ids}")
    print(f"总共 {len(all_particle_ids)} 个粒子")

    # 输入要绘制的粒子ID
    particle_ids_input = input("请输入要绘制的粒子ID（用逗号分隔，直接回车表示使用所有粒子）：")

    if particle_ids_input.strip() == "":
        # 如果用户直接回车，使用所有粒子
        particle_ids = all_particle_ids
        print(f"将使用所有 {len(particle_ids)} 个粒子进行分析")
    else:
        # 解析用户输入的粒子ID
        try:
            particle_ids = list(map(int, particle_ids_input.split(',')))
            # 验证输入的ID是否存在于文件中
            invalid_ids = [pid for pid in particle_ids if pid not in all_particle_ids]
            if invalid_ids:
                print(f"警告: 以下粒子ID在文件中不存在: {invalid_ids}")
                particle_ids = [pid for pid in particle_ids if pid in all_particle_ids]

            if not particle_ids:
                print("没有有效的粒子ID，将使用所有粒子")
                particle_ids = all_particle_ids
        except ValueError:
            print("输入格式错误，将使用所有粒子")
            particle_ids = all_particle_ids

    # 设置默认保存路径
    default_save_path = "../Result"
    os.makedirs(default_save_path, exist_ok=True)

    # 读取粒子的速度和位置信息
    print(f"正在读取 {len(particle_ids)} 个粒子的数据...")
    particle_data = read_p4p_file(p4p_file_name, particle_ids)

    # 读取接触力信息
    #contact_data = read_p4c_file(p4c_file_name)

    # 询问用户是否要提取特定数据
    extract_specific = input("是否要提取特定粒子的特定数据？(y/n): ")
    if extract_specific.lower() == 'y':
        # 获取要提取的粒子ID
        extract_pid_input = input("请输入要提取数据的粒子ID（用逗号分隔，留空表示所有粒子）：")
        extract_pids = None if extract_pid_input.strip() == "" else list(map(int, extract_pid_input.split(',')))

        # 获取要提取的数据类型
        print("可用的数据类型: time, vx, vy,vz, px, py, pz")
        extract_type_input = input("请输入要提取的数据类型（用逗号分隔，留空表示所有类型）：")
        extract_types = None if extract_type_input.strip() == "" else extract_type_input.split(',')

        # 提取并导出数据
        output_filename = input(
            "请输入输出文件名（默认为'extracted_specific_data.csv'）：") or "extracted_specific_data.csv"
        output_path = os.path.join(default_save_path, output_filename)
        export_specific_data_to_csv(particle_data, output_path, extract_pids, extract_types)

    # 询问用户是否要绘制图形
    draw_plots = input("是否绘制图形？(y/n): ")
    if draw_plots.lower() == 'y':

        # 绘制速度与时间的关系图
        plot_speed_vs_time(particle_data, default_save_path)

        # 绘制位移与时间的关系图
        plot_displacement_vs_time(particle_data, default_save_path)

        plot_physical_distance_vs_time(particle_data, default_save_path)
        # 绘制接触力与位移的关系图
        # plot_force_vs_displacement(particle_data, contact_data, default_save_path)
        #plot_fluid_force_vs_time(particle_data, default_save_path)

        # 在main函数中添加：
        # plot_force_vs_time(particle_data, contact_data, default_save_path)

        # 绘制动能图（总是绘制，因为是所有粒子的总和）
        #print("正在计算和绘制动能变化...")
        #plot_kinetic_energy_vs_time(particle_data, default_save_path)

    print("所有操作已完成！")


if __name__ == "__main__":
    main()