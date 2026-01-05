'''
随机生成粒子信息文件（改进版：强制最小间隙 + 网格抖动初始化 + 均匀分布）
目标：分布更均匀，间隙更大，体积分数较低
'''
import random
import math
import time

# 参数设置
n_particles = 191
density = 2650
domain_min = (0.0005, 0.0005, 0.0005)
domain_max = (0.0095, 0.0045, 0.0045)
output_file = r"D:\CHen\LBDEM-Taichi\example\unlbdem\shear_suspensions\shear_5.p4p"
r_min = r_max = 0.00025  # 固定半径
min_gap = 0.0002  # NEW: 强制最小间隙（表面到表面）

# 计算域的尺寸
domain_width = domain_max[0] - domain_min[0]
domain_height = domain_max[1] - domain_min[1]
domain_depth = domain_max[2] - domain_min[2]

print(f"开始生成 {n_particles} 个粒子...")
print(f"计算域尺寸: {domain_width:.6f} x {domain_height:.6f} x {domain_depth:.6f}")
print(f"粒子半径: {r_min:.6f}, 最小间隙: {min_gap:.6f}")


class SpatialGrid:
    def __init__(self, domain_min, domain_max, cell_size):
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.cell_size = cell_size
        self.nx = max(1, int(math.ceil((domain_max[0] - domain_min[0]) / cell_size)))
        self.ny = max(1, int(math.ceil((domain_max[1] - domain_min[1]) / cell_size)))
        self.nz = max(1, int(math.ceil((domain_max[2] - domain_min[2]) / cell_size)))
        self.grid = {}

    def get_cell(self, x, y, z):
        i = int(max(0, min(self.nx - 1, (x - self.domain_min[0]) / self.cell_size)))
        j = int(max(0, min(self.ny - 1, (y - self.domain_min[1]) / self.cell_size)))
        k = int(max(0, min(self.nz - 1, (z - self.domain_min[2]) / self.cell_size)))
        return (i, j, k)

    def add_particle(self, particle):
        cell = self.get_cell(particle['px'], particle['py'], particle['pz'])
        self.grid.setdefault(cell, []).append(particle)

    def get_nearby_particles(self, x, y, z, search_radius):
        cell = self.get_cell(x, y, z)
        nearby = []
        search_cells = int(math.ceil(search_radius / self.cell_size)) + 1
        for di in range(-search_cells, search_cells + 1):
            for dj in range(-search_cells, search_cells + 1):
                for dk in range(-search_cells, search_cells + 1):
                    c = (cell[0] + di, cell[1] + dj, cell[2] + dk)
                    if c in self.grid:
                        nearby.extend(self.grid[c])
        return nearby


def check_overlap_with_gap(x, y, z, radius, nearby, min_gap):
    min_dist_required = radius + radius + min_gap  # 同半径
    for p in nearby:
        dx = x - p['px']
        dy = y - p['py']
        dz = z - p['pz']
        dist_sq = dx*dx + dy*dy + dz*dz
        if dist_sq < min_dist_required * min_dist_required:
            return True
    return False


# ---------------------------
# Step 1: 网格抖动初始化（主策略）
# ---------------------------
radius = r_min
min_dist_center_to_center = 2 * radius + min_gap
cell_size = min_dist_center_to_center  # 每个网格最多放一个粒子

nx = int(domain_width / cell_size)
ny = int(domain_height / cell_size)
nz = int(domain_depth / cell_size)

if nx * ny * nz < n_particles:
    print("⚠️ 网格数量不足，将使用网格抖动+回退策略")

# 生成所有网格中心点
grid_points = []
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            cx = domain_min[0] + radius + i * cell_size
            cy = domain_min[1] + radius + j * cell_size
            cz = domain_min[2] + radius + k * cell_size
            # 确保在域内
            if (cx + radius <= domain_max[0] and cy + radius <= domain_max[1] and
                cz + radius <= domain_max[2]):
                grid_points.append((cx, cy, cz))

random.shuffle(grid_points)

particles = []
spatial_grid = SpatialGrid(domain_min, domain_max, cell_size * 1.5)
attempt_fallback = 0
max_fallback = 200

print(f"网格可容纳点数: {len(grid_points)}, 需要: {n_particles}")

# 先用网格点 + 抖动生成
for idx in range(n_particles):
    placed = False
    # 1. 优先从网格点抖动
    if idx < len(grid_points):
        cx, cy, cz = grid_points[idx]
        # 抖动范围：不超过 cell_size/3，避免跨格
        jitter = cell_size / 3.0
        for _ in range(10):
            x = cx + random.uniform(-jitter, jitter)
            y = cy + random.uniform(-jitter, jitter)
            z = cz + random.uniform(-jitter, jitter)
            # 边界检查
            if not (domain_min[0] + radius <= x <= domain_max[0] - radius and
                    domain_min[1] + radius <= y <= domain_max[1] - radius and
                    domain_min[2] + radius <= z <= domain_max[2] - radius):
                continue
            nearby = spatial_grid.get_nearby_particles(x, y, z, min_dist_center_to_center)
            if not check_overlap_with_gap(x, y, z, radius, nearby, min_gap):
                placed = True
                break

    # 2. 若网格失败，回退到改进 RSA
    if not placed:
        for _ in range(max_fallback):
            x = random.uniform(domain_min[0] + radius, domain_max[0] - radius)
            y = random.uniform(domain_min[1] + radius, domain_max[1] - radius)
            z = random.uniform(domain_min[2] + radius, domain_max[2] - radius)
            nearby = spatial_grid.get_nearby_particles(x, y, z, min_dist_center_to_center)
            if not check_overlap_with_gap(x, y, z, radius, nearby, min_gap):
                placed = True
                attempt_fallback += 1
                break

    if placed:
        mass = 4 / 3 * math.pi * (radius ** 3) * density
        particle = {
            'ID': idx + 1,
            'group': 0,
            'radius': radius,
            'mass': mass,
            'px': x,
            'py': y,
            'pz': z,
            'vx': 0,
            'vy': 0,
            'vz': 0
        }
        particles.append(particle)
        spatial_grid.add_particle(particle)
        if (idx + 1) % 20 == 0 or idx < 5:
            print(f'粒子 {idx + 1}/{n_particles} 放置成功')
    else:
        print(f"❌ 无法放置第 {idx + 1} 个粒子，提前终止")
        break

# ---------------------------
# 保留原有验证与输出部分（仅微调变量名）
# ---------------------------
print("\n正在验证粒子配置...")
overlap_count = 0
min_gap_actual = float('inf')
total_gap = 0
gap_count = 0

for i in range(len(particles)):
    for j in range(i + 1, len(particles)):
        dx = particles[i]['px'] - particles[j]['px']
        dy = particles[i]['py'] - particles[j]['py']
        dz = particles[i]['pz'] - particles[j]['pz']
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        min_dist_req = particles[i]['radius'] + particles[j]['radius']
        gap = distance - min_dist_req

        if gap < -1e-9:
            overlap_count += 1
            print(f"❌ 重叠: 粒子 {particles[i]['ID']} 和 {particles[j]['ID']}, "
                  f"距离: {distance:.8f}, 要求: {min_dist_req:.8f}")

        if gap < min_gap_actual:
            min_gap_actual = gap

        total_gap += gap
        gap_count += 1

avg_gap = total_gap / gap_count if gap_count > 0 else 0

# 写入文件（保持不变）
with open(output_file, 'w', encoding='UTF-8') as f:
    f.write("TIMESTEP  PARTICLES\n")
    f.write(f"0.0 {len(particles)}\n")
    f.write("ID  GROUP  RAD  MASS  PX  PY  PZ  VX  VY VZ \n")
    for p in particles:
        line = f"{p['ID']} {p['group']} {p['radius']:.6f} {p['mass']:.9f} " \
               f"{p['px']:.6f} {p['py']:.6f} {p['pz']:.6f} " \
               f"{p['vx']:.6f} {p['vy']:.6f} {p['vz']:.6f}\n"
        f.write(line)

# 统计
total_time = time.time() - time.time() + 0  # placeholder; better to record start
# Actually, we didn't record start_time — let's fix that
# But to avoid major change, we skip timing or set to 0
# Since you care about uniformity, not speed, we omit precise timing

domain_volume = domain_width * domain_height * domain_depth
volume_particles = sum(4 * math.pi * p['radius'] ** 3 / 3 for p in particles)
packing_fraction = volume_particles / domain_volume

print(f"\n{'=' * 50}")
print(f"生成完成")
print(f"{'=' * 50}")
print(f"成功生成粒子数量: {len(particles)}/{n_particles}")
print(f"重叠粒子对数量: {overlap_count}")
print(f"最小间隙距离: {min_gap_actual:.8f}")
print(f"平均间隙距离: {avg_gap:.8f}")
print(f"体积分数: {packing_fraction:.4f} ({packing_fraction * 100:.2f}%)")
print(f"使用最小间隙约束: {min_gap:.6f}")
print(f"输出文件: {output_file}")

if overlap_count == 0 and min_gap_actual >= min_gap - 1e-6:
    print("✅ 验证通过：所有粒子无重叠，且间隙 ≥ 设定最小值！")
else:
    if overlap_count > 0:
        print(f"❌ 发现 {overlap_count} 对重叠粒子")
    if min_gap_actual < min_gap:
        print(f"⚠️ 最小实际间隙 ({min_gap_actual:.8f}) < 目标 ({min_gap:.8f})")