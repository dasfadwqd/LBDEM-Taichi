import os
import pickle
import glob
from example.VTK.vtkWriter3D import VtkWriter3D  # 确保这个文件在你的 Python 路径中

# 参数设置
outDir = 'D:\CHen\LBM-DEM-3d\example\lbdem\single_particle_settle\Re1.5_fully'  # 替换为你的 outDir
results_dir = os.path.join(outDir, 'results')
vtk_dir = os.path.join(outDir, 'vtk')  # 输出 vtk 文件夹

# 获取所有结果文件
result_files = sorted(glob.glob(os.path.join(results_dir, 'result_*.dat')))

with open(result_files[0], 'rb') as fid:
    sample = pickle.load(fid)
velf = sample['velf']  # shape: (nx, ny, nz, 3)
nx, ny, nz = velf.shape[:3]
# 创建 VTK writer
writer = VtkWriter3D(nx=nx, ny=ny, nz=nz, outputDir=outDir, subDir='vtk')

# 遍历所有时间步
for step_idx, filepath in enumerate(result_files):
    with open(filepath, 'rb') as fid:
        data = pickle.load(fid)

    velf = data['velf']  # shape: (nx, ny, nz, 3)
    rhof = data['rhof']  # shape: (nx, ny, nz)
    pf = data['pf']      # shape: (nx, ny, nz)

    # 添加矢量场
    writer.addVectorField(velf, 'Velocity')

    # 添加标量场
    writer.addScalarField(rhof, 'Density')
    writer.addScalarField(pf, 'Pressure')

    # 写入 VTK 文件
    writer.write(step_idx)

    # 清空字段，准备下一个时间步
    writer.clearFields()

print("✅ 所有结果已转换为 VTK 格式，保存在：", vtk_dir)