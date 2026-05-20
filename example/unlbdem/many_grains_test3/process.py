#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析颗粒文件:
1. 从many_grains.p4p读取t=0时刻的数据,锁定顶部10%的颗粒ID
2. 从output.p4p读取所有时间步,跟踪这些颗粒的平均VZ速度

配置说明:
---------
可以通过以下方式修改百分比:

方法1: 修改下面的默认配置
DEFAULT_PERCENTAGE = 10  # 修改这个数值来改变默认百分比

方法2: 通过命令行参数
python3 analyze_two_files.py many_grains.p4p output.p4p 20  # 使用20%

方法3: 直接修改main()函数中的 percentage = 10
"""

import numpy as np
import sys
import os

# ============= 配置区域 =============
DEFAULT_PERCENTAGE = 2  # 默认跟踪顶部的百分比 (可修改此值: 1-100)


# ===================================

def read_single_timestep_file(filename):
    """
    读取单个时间步的颗粒文件 (many_grains.p4p)

    参数:
        filename: 颗粒文件路径

    返回:
        particles: 颗粒数据数组
        timestep: 时间步
        num_particles: 颗粒数量
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 第一行: TIMESTEP PARTICLES (标题)
    # 第二行: 0.0 4584 (实际数据)
    # 第三行: ID GROUP RAD ... (列标题)
    # 从第四行开始是数据

    # 读取时间步和颗粒数量 (第二行)
    data_header = lines[1].strip().split()
    timestep = float(data_header[0])
    num_particles = int(data_header[1])

    # 跳过前三行
    data_lines = lines[3:]

    # 解析颗粒数据
    particles = []
    for line in data_lines:
        if line.strip():  # 跳过空行
            values = line.strip().split()
            if len(values) >= 10:  # 确保有足够的列
                particles.append([float(v) for v in values])

    return np.array(particles), timestep, num_particles


def read_multi_timestep_file(filename):
    """
    读取包含多个时间步的颗粒文件 (output.p4p)

    参数:
        filename: 颗粒文件路径

    返回:
        timesteps: 时间步列表
        all_particles: 字典,键为时间步,值为该时间步的颗粒数据数组
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    timesteps = []
    all_particles = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 跳过空行
        if not line:
            i += 1
            continue

        # 检查是否是时间步标题行
        if line.startswith('TIMESTEP') or line.split()[0].replace('.', '', 1).replace('-', '', 1).isdigit():
            # 读取时间步和颗粒数
            parts = line.split()
            if parts[0] == 'TIMESTEP':
                # 这是标题行,跳到下一行读取实际数据
                i += 1
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()

            timestep = float(parts[0])
            num_particles = int(parts[1])

            timesteps.append(timestep)

            # 跳过列标题行
            i += 1
            if i < len(lines) and 'ID' in lines[i]:
                i += 1

            # 读取该时间步的所有颗粒数据
            particles = []
            count = 0
            while i < len(lines) and count < num_particles:
                line = lines[i].strip()
                if line and not line.startswith('TIMESTEP') and 'ID' not in line:
                    values = line.split()
                    if len(values) >= 10:  # 确保有足够的列
                        particles.append([float(v) for v in values])
                        count += 1
                i += 1

            all_particles[timestep] = np.array(particles)
        else:
            i += 1

    return timesteps, all_particles


def get_top_particle_ids(particles, percentage=10):
    """
    根据Z位置(PZ)获取顶部指定百分比的颗粒ID

    参数:
        particles: 颗粒数据数组
        percentage: 要选择的顶部百分比

    返回:
        top_ids: 顶部颗粒的ID列表
        top_pz: 顶部颗粒的PZ值
    """
    # 提取ID (索引0) 和 PZ (位置Z坐标, 索引6)
    ids = particles[:, 0].astype(int)
    pz = particles[:, 6]

    # 按Z位置排序,找出顶部的颗粒
    sorted_indices = np.argsort(pz)[::-1]  # 降序排列

    # 计算顶部百分比的颗粒数量
    num_top = int(len(particles) * percentage / 100)
    if num_top < 1:
        num_top = 1

    # 获取顶部颗粒的ID
    top_indices = sorted_indices[:num_top]
    top_ids = ids[top_indices]

    return top_ids, pz[top_indices]


def get_particles_by_ids(particles, target_ids):
    """
    根据ID列表提取特定颗粒的数据

    参数:
        particles: 颗粒数据数组
        target_ids: 目标颗粒ID列表

    返回:
        selected_particles: 选中的颗粒数据
        found_ids: 实际找到的颗粒ID
    """
    ids = particles[:, 0].astype(int)
    selected_particles = []
    found_ids = []

    for target_id in target_ids:
        mask = ids == target_id
        if np.any(mask):
            selected_particles.append(particles[mask][0])
            found_ids.append(target_id)

    if len(selected_particles) > 0:
        return np.array(selected_particles), found_ids
    else:
        return np.array([]), []


def main():
    # 默认文件名和百分比
    initial_file = r'D:\CHen\LBDEM-Taichi\example\unlbdem\many_grains_test2\20grains.p4p'  # t=0时刻的文件
    output_file = r'D:\CHen\LBDEM-Taichi\example\unlbdem\many_grains_test2\test20\output.p4p'  # 多时间步的文件
    percentage = DEFAULT_PERCENTAGE  # 使用配置区域的默认值

    # 允许通过命令行参数指定文件和百分比
    # 用法: python3 script.py [initial_file] [output_file] [percentage]
    if len(sys.argv) > 1:
        initial_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3:
        try:
            percentage = float(sys.argv[3])
            if percentage <= 0 or percentage > 100:
                print(f"警告: 百分比应在0-100之间,使用默认值{DEFAULT_PERCENTAGE}%")
                percentage = DEFAULT_PERCENTAGE
        except ValueError:
            print(f"警告: 无效的百分比值,使用默认值{DEFAULT_PERCENTAGE}%")
            percentage = DEFAULT_PERCENTAGE

    try:
        # ===== 步骤1: 从many_grains.p4p读取t=0时刻数据,锁定顶部颗粒 =====
        print(f"步骤1: 读取初始文件 {initial_file}")

        if not os.path.exists(initial_file):
            print(f"错误: 找不到文件 '{initial_file}'")
            return

        initial_particles, initial_timestep, initial_num = read_single_timestep_file(initial_file)

        print(f"  时间步: {initial_timestep}")
        print(f"  颗粒数: {initial_num}")

        # 获取顶部指定百分比的颗粒ID
        top_ids, top_pz = get_top_particle_ids(initial_particles, percentage)

        print(f"\n=== 锁定的顶部 {percentage}% 颗粒 ===")
        print(f"颗粒数量: {len(top_ids)}")
        print(f"颗粒ID: {[int(x) for x in sorted(top_ids)]}")
        print(f"初始Z位置范围: [{np.min(top_pz):.6f}, {np.max(top_pz):.6f}]")

        # ===== 步骤2: 从output.p4p读取所有时间步,跟踪这些颗粒 =====
        print(f"\n步骤2: 读取输出文件 {output_file}")

        if not os.path.exists(output_file):
            print(f"错误: 找不到文件 '{output_file}'")
            return

        timesteps, all_particles = read_multi_timestep_file(output_file)

        print(f"  读取到 {len(timesteps)} 个时间步")
        if len(timesteps) > 0:
            print(f"  时间步范围: {min(timesteps)} 到 {max(timesteps)}")

        if len(timesteps) == 0:
            print("错误: 未找到任何时间步数据")
            return

        # ===== 步骤3: 对每个时间步,跟踪锁定颗粒的平均VZ和平均PZ =====
        print(f"\n=== 各时间步的跟踪颗粒数据 ===")
        print(f"{'时间步':<12} {'平均PZ(高度)':<15} {'平均VZ':<15} {'颗粒数':<10} {'PZ范围':<25} {'VZ范围'}")
        print("-" * 100)

        results = []
        for ts in sorted(timesteps):
            particles = all_particles[ts]
            selected_particles, found_ids = get_particles_by_ids(particles, top_ids)

            if len(selected_particles) > 0:
                pz_values = selected_particles[:, 6]  # PZ在索引6
                vz_values = selected_particles[:, 9]  # VZ在索引9

                avg_pz = np.mean(pz_values)
                avg_vz = np.mean(vz_values)
                min_pz = np.min(pz_values)
                max_pz = np.max(pz_values)
                min_vz = np.min(vz_values)
                max_vz = np.max(vz_values)

                results.append({
                    'timestep': ts,
                    'avg_pz': avg_pz,
                    'avg_vz': avg_vz,
                    'num_found': len(found_ids),
                    'min_pz': min_pz,
                    'max_pz': max_pz,
                    'min_vz': min_vz,
                    'max_vz': max_vz
                })

                print(
                    f"{ts:<12.6f} {avg_pz:<15.6f} {avg_vz:<15.6f} {len(found_ids):<10} [{min_pz:.6f}, {max_pz:.6f}] [{min_vz:.6f}, {max_vz:.6f}]")
            else:
                print(f"{ts:<12.6f} {'N/A':<15} {'N/A':<15} {0:<10} 未找到跟踪颗粒")

        # ===== 步骤4: 总体统计 =====
        if len(results) > 0:
            all_avg_pz = [r['avg_pz'] for r in results]
            all_avg_vz = [r['avg_vz'] for r in results]
            print(f"\n=== 总体统计 ===")
            print(f"时间步数: {len(results)}")
            print(f"\n高度统计 (PZ):")
            print(f"  平均PZ的总体平均值: {np.mean(all_avg_pz):.6f}")
            print(f"  平均PZ的标准差: {np.std(all_avg_pz):.6f}")
            print(f"  平均PZ的范围: [{np.min(all_avg_pz):.6f}, {np.max(all_avg_pz):.6f}]")
            print(f"\nZ方向速度统计 (VZ):")
            print(f"  平均VZ的总体平均值: {np.mean(all_avg_vz):.6f}")
            print(f"  平均VZ的标准差: {np.std(all_avg_vz):.6f}")
            print(f"  平均VZ的范围: [{np.min(all_avg_vz):.6f}, {np.max(all_avg_vz):.6f}]")

        # ===== 步骤5: 保存结果到文件 =====
        result_file = 'tracked_particles_vz.txt'
        with open(result_file, 'w') as f:
            f.write(f"顶部{percentage}%颗粒的跟踪分析结果\n")
            f.write(f"=" * 70 + "\n\n")
            f.write(f"初始文件: {initial_file}\n")
            f.write(f"输出文件: {output_file}\n\n")
            f.write(f"锁定的颗粒ID: {[int(x) for x in sorted(top_ids)]}\n")
            f.write(f"颗粒数量: {len(top_ids)}\n")
            f.write(f"初始时间步: {initial_timestep}\n")
            f.write(f"初始Z位置范围: [{np.min(top_pz):.6f}, {np.max(top_pz):.6f}]\n\n")
            f.write(f"{'时间步':<12} {'平均PZ':<15} {'平均VZ':<15} {'颗粒数':<10}\n")
            f.write("-" * 55 + "\n")
            for r in results:
                f.write(f"{r['timestep']:<12.6f} {r['avg_pz']:<15.6f} {r['avg_vz']:<15.6f} {r['num_found']:<10}\n")

            if len(results) > 0:
                f.write("\n" + "=" * 70 + "\n")
                f.write("总体统计\n")
                f.write("=" * 70 + "\n")
                f.write(f"时间步数: {len(results)}\n\n")
                f.write("高度统计 (PZ):\n")
                f.write(f"  平均PZ的总体平均值: {np.mean(all_avg_pz):.6f}\n")
                f.write(f"  平均PZ的标准差: {np.std(all_avg_pz):.6f}\n")
                f.write(f"  平均PZ的范围: [{np.min(all_avg_pz):.6f}, {np.max(all_avg_pz):.6f}]\n\n")
                f.write("Z方向速度统计 (VZ):\n")
                f.write(f"  平均VZ的总体平均值: {np.mean(all_avg_vz):.6f}\n")
                f.write(f"  平均VZ的标准差: {np.std(all_avg_vz):.6f}\n")
                f.write(f"  平均VZ的范围: [{np.min(all_avg_vz):.6f}, {np.max(all_avg_vz):.6f}]\n")

        print(f"\n结果已保存到: {result_file}")

    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e}")
        print("请确保文件在当前目录,或提供完整路径")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 70)
    print("颗粒跟踪分析程序")
    print("=" * 70)
    print("用法: python3 analyze_two_files.py [initial_file] [output_file] [percentage]")
    print(f"默认: many_grains.p4p, output.p4p, {DEFAULT_PERCENTAGE}%")
    print("示例: python3 analyze_two_files.py data.p4p output.p4p 15")
    print("=" * 70)
    print()
    main()