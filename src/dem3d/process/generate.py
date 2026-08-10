'''
随机生成粒子信息文件（改进版：目标间距均匀排布 + 随机抖动 + 最小间隙约束）
策略：直接根据目标体积分数计算最优间距，生成规则晶格后施加随机扰动，最后随机填补空隙
'''
import random
import math
import time
from collections import defaultdict
# 275, 458,871,1283, 1604


# ==================== 参数设置 ====================
n_particles = 1283          # 目标粒子数（改这个数字就能控制体积分数）
density = 1120
a = 5e-4
domain_min = (0.006, 0.0+a, 0.0+a)
domain_max = (0.054, 0.02-a, 0.02-a)
output_file = r"D:\CHen\LBDEM-Taichi\example\unlbdem\pressure_drop_un\vf0.28.p4p"
r_min = r_max = 0.001      # 固定半径
min_gap = 2.8e-4             # 强制最小间隙（从1e-3降低，释放高体积分数空间）

# 抖动比例 (0.0~0.5)：控制粒子在网格内的随机偏移幅度
jitter_ratio = 0.4

# ==================== 计算域信息 ====================
domain_width  = domain_max[0] - domain_min[0]
domain_height = domain_max[1] - domain_min[1]
domain_depth  = domain_max[2] - domain_min[2]
domain_volume = domain_width * domain_height * domain_depth
particle_volume = 4.0/3.0 * math.pi * r_min**3

# 预估体积分数
est_vf = n_particles * particle_volume / domain_volume

print(f"计算域尺寸: {domain_width:.6f} x {domain_height:.6f} x {domain_depth:.6f}")
print(f"粒子半径: {r_min:.6f}, 最小间隙: {min_gap:.6f}")
print(f"目标粒子数: {n_particles}")
print(f"预估体积分数: {est_vf:.4f} ({est_vf*100:.2f}%)")
print(f"=" * 60)


# ==================== 空间网格加速 ====================
class SpatialGrid:
    def __init__(self, domain_min, domain_max, cell_size):
        self.domain_min = domain_min
        self.cell_size = cell_size
        self.nx = max(1, int(math.ceil((domain_max[0] - domain_min[0]) / cell_size)))
        self.ny = max(1, int(math.ceil((domain_max[1] - domain_min[1]) / cell_size)))
        self.nz = max(1, int(math.ceil((domain_max[2] - domain_min[2]) / cell_size)))
        self.grid = {}

    def _cell(self, x, y, z):
        i = int(max(0, min(self.nx - 1, (x - self.domain_min[0]) / self.cell_size)))
        j = int(max(0, min(self.ny - 1, (y - self.domain_min[1]) / self.cell_size)))
        k = int(max(0, min(self.nz - 1, (z - self.domain_min[2]) / self.cell_size)))
        return (i, j, k)

    def add(self, p):
        self.grid.setdefault(self._cell(p['px'], p['py'], p['pz']), []).append(p)

    def nearby(self, x, y, z, radius):
        c = self._cell(x, y, z)
        n = int(math.ceil(radius / self.cell_size)) + 1
        result = []
        for di in range(-n, n+1):
            for dj in range(-n, n+1):
                for dk in range(-n, n+1):
                    key = (c[0]+di, c[1]+dj, c[2]+dk)
                    if key in self.grid:
                        result.extend(self.grid[key])
        return result


def has_conflict(x, y, z, nearby_list, min_dist):
    """检查是否与附近粒子冲突（距离 < min_dist）"""
    min_sq = min_dist * min_dist
    for p in nearby_list:
        dx, dy, dz = x - p['px'], y - p['py'], z - p['pz']
        if dx*dx + dy*dy + dz*dz < min_sq:
            return True
    return False


def make_particle(pid, x, y, z):
    return {
        'ID': pid, 'group': 0, 'radius': r_min,
        'mass': particle_volume * density,
        'px': x, 'py': y, 'pz': z,
        'vx': 0, 'vy': 0, 'vz': 0
    }


# ==================== 阶段1：抖动晶格 ====================
def generate_jittered_lattice(spacing):
    """
    在给定间距下生成规则晶格 + 随机抖动
    抖动范围确保不会导致相邻网格粒子重叠
    """
    radius = r_min
    nx = max(1, int((domain_width  - 2*radius) / spacing) + 1)
    ny = max(1, int((domain_height - 2*radius) / spacing) + 1)
    nz = max(1, int((domain_depth  - 2*radius) / spacing) + 1)

    if nx == 1:
        xs = [domain_min[0] + domain_width / 2]
    else:
        xs = [domain_min[0] + radius + i * (domain_width - 2*radius) / (nx - 1) for i in range(nx)]
    if ny == 1:
        ys = [domain_min[1] + domain_height / 2]
    else:
        ys = [domain_min[1] + radius + j * (domain_height - 2*radius) / (ny - 1) for j in range(ny)]
    if nz == 1:
        zs = [domain_min[2] + domain_depth / 2]
    else:
        zs = [domain_min[2] + radius + k * (domain_depth - 2*radius) / (nz - 1) for k in range(nz)]

    # 安全抖动量：不超过 (间距 - 最小中心距) / 2
    min_center = 2 * radius + min_gap
    max_jitter = max(0.0, (spacing - min_center) / 2.0) * jitter_ratio

    positions = []
    for x0 in xs:
        for y0 in ys:
            for z0 in zs:
                # 随机抖动（尝试5次找合规位置）
                for _ in range(5):
                    x = x0 + random.uniform(-max_jitter, max_jitter)
                    y = y0 + random.uniform(-max_jitter, max_jitter)
                    z = z0 + random.uniform(-max_jitter, max_jitter)
                    # 边界检查
                    if (domain_min[0] + radius <= x <= domain_max[0] - radius and
                        domain_min[1] + radius <= y <= domain_max[1] - radius and
                        domain_min[2] + radius <= z <= domain_max[2] - radius):
                        positions.append((x, y, z))
                        break
                else:
                    # 抖动失败，用原始坐标
                    positions.append((x0, y0, z0))
    return positions


# ==================== 主流程 ====================
particles = []
particle_id = 0
min_center_dist = 2 * r_min + min_gap

# --- 计算最优晶格间距 ---
# 目标：让晶格刚好能放下 ~n_particles 个点
# 体积 V = nx*ny*nz * spacing^3 ≈ n_particles * spacing^3
# 但要考虑边界留白，用有效体积近似
eff_vol = (domain_width - 2*r_min) * (domain_height - 2*r_min) * (domain_depth - 2*r_min)
ideal_spacing = (eff_vol / n_particles) ** (1.0/3.0)
ideal_spacing = max(min_center_dist + 1e-6, ideal_spacing)

print(f"最优晶格间距: {ideal_spacing:.6f}")

# --- 阶段1：抖动晶格 ---
spatial_grid = SpatialGrid(domain_min, domain_max, min_center_dist * 1.2)
lattice_positions = generate_jittered_lattice(ideal_spacing)
random.shuffle(lattice_positions)  # 打乱顺序，消除方向偏好

added_lattice = 0
for (x, y, z) in lattice_positions:
    if len(particles) >= n_particles:
        break
    nearby = spatial_grid.nearby(x, y, z, min_center_dist)
    if not has_conflict(x, y, z, nearby, min_center_dist):
        particle_id += 1
        p = make_particle(particle_id, x, y, z)
        particles.append(p)
        spatial_grid.add(p)
        added_lattice += 1

vf_now = len(particles) * particle_volume / domain_volume
print(f"阶段1完成（抖动晶格）: {added_lattice} 个, 总计 {len(particles)}/{n_particles}, vf={vf_now:.4f}")

# --- 阶段2：随机补足 ---
if len(particles) < n_particles:
    remaining = n_particles - len(particles)
    print(f"阶段2（随机补足）: 需要再放 {remaining} 个...")
    attempts = 0
    max_attempts = 500000
    report_interval = max(50, n_particles // 10)

    while len(particles) < n_particles and attempts < max_attempts:
        attempts += 1
        x = random.uniform(domain_min[0] + r_min, domain_max[0] - r_min)
        y = random.uniform(domain_min[1] + r_min, domain_max[1] - r_min)
        z = random.uniform(domain_min[2] + r_min, domain_max[2] - r_min)
        nearby = spatial_grid.nearby(x, y, z, min_center_dist)
        if not has_conflict(x, y, z, nearby, min_center_dist):
            particle_id += 1
            p = make_particle(particle_id, x, y, z)
            particles.append(p)
            spatial_grid.add(p)
            if len(particles) % report_interval == 0:
                vf_tmp = len(particles) * particle_volume / domain_volume
                print(f"  进度: {len(particles)}/{n_particles}, vf={vf_tmp:.4f}, 尝试={attempts}")

    if len(particles) < n_particles:
        print(f"⚠️ 随机补足结束，最终 {len(particles)}/{n_particles} (尝试 {attempts} 次)")
    else:
        print(f"✅ 随机补足完成，总计 {len(particles)} 个粒子")


# ==================== 验证 ====================
print(f"\n{'=' * 60}")
print("验证中...")
print(f"{'=' * 60}")

if len(particles) == 0:
    print("❌ 未生成任何粒子！")
    exit()

overlap_count = 0
min_gap_actual = float('inf')
gap_distribution = []
nearest_gaps = []

for i in range(len(particles)):
    min_gap_i = float('inf')
    for j in range(len(particles)):
        if i == j:
            continue
        dx = particles[i]['px'] - particles[j]['px']
        dy = particles[i]['py'] - particles[j]['py']
        dz = particles[i]['pz'] - particles[j]['pz']
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        gap = dist - 2 * r_min
        if j > i:
            if gap < -1e-9:
                overlap_count += 1
            else:
                gap_distribution.append(gap)
            if gap < min_gap_actual:
                min_gap_actual = gap
        if gap < min_gap_i:
            min_gap_i = gap
    nearest_gaps.append(min_gap_i)

# ==================== 输出文件 ====================
with open(output_file, 'w', encoding='UTF-8') as f:
    f.write("TIMESTEP  PARTICLES\n")
    f.write(f"0.0 {len(particles)}\n")
    f.write("ID  GROUP  RAD  MASS  PX  PY  PZ  VX  VY VZ \n")
    for p in particles:
        f.write(f"{p['ID']} {p['group']} {p['radius']:.6f} {p['mass']:.9f} "
                f"{p['px']:.6f} {p['py']:.6f} {p['pz']:.6f} "
                f"{p['vx']:.6f} {p['vy']:.6f} {p['vz']:.6f}\n")

# ==================== 统计报告 ====================
final_vf = len(particles) * particle_volume / domain_volume

print(f"成功生成粒子数量: {len(particles)}/{n_particles}")
print(f"重叠粒子对数量: {overlap_count}")
print(f"最小间隙距离: {min_gap_actual:.8f}")

if gap_distribution:
    gap_distribution.sort()
    avg_gap = sum(gap_distribution) / len(gap_distribution)
    print(f"平均间隙距离: {avg_gap:.8f}")
    print(f"间隙分布 [min, 25%, 中位, 75%, max]: "
          f"[{gap_distribution[0]:.6f}, {gap_distribution[len(gap_distribution)//4]:.6f}, "
          f"{gap_distribution[len(gap_distribution)//2]:.6f}, "
          f"{gap_distribution[3*len(gap_distribution)//4]:.6f}, "
          f"{gap_distribution[-1]:.6f}]")

if nearest_gaps:
    avg_ng = sum(nearest_gaps) / len(nearest_gaps)
    std_ng = (sum((g - avg_ng)**2 for g in nearest_gaps) / len(nearest_gaps)) ** 0.5
    print(f"最近邻平均间隙: {avg_ng:.8f}")
    print(f"最近邻间隙标准差: {std_ng:.8f}")

print(f"体积分数: {final_vf:.4f} ({final_vf * 100:.2f}%)")
print(f"输出文件: {output_file}")

if overlap_count == 0 and min_gap_actual >= min_gap - 1e-6:
    print("✅ 验证通过：所有粒子无重叠，且间隙 ≥ 设定最小值！")
else:
    if overlap_count > 0:
        print(f"❌ 发现 {overlap_count} 对重叠粒子")
    if min_gap_actual < min_gap:
        print(f"⚠️ 最小实际间隙 ({min_gap_actual:.8f}) < 目标 ({min_gap:.8f})")

