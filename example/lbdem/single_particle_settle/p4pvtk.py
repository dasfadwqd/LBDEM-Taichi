import vtk
import os


def p4p_to_vtp_series(p4p_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(p4p_path, 'r') as f:
        content = f.read()

    blocks = content.strip().split('TIMESTEP  PARTICLES')
    timestep_index = 0

    for block in blocks[1:]:
        lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
        if len(lines) < 3:
            continue

        # ✅ 正确拆分时间和粒子数
        time_val, n_particles_str = lines[0].split()
        time_val = float(time_val)
        n_particles = int(n_particles_str)

        # 表头在 lines[1]，数据从 lines[2] 开始
        data_start = 2
        if lines[1].startswith('ID'):
            data_start = 2  # 跳过表头

        points = vtk.vtkPoints()
        radius_arr = vtk.vtkDoubleArray()
        radius_arr.SetName("Radius")
        velocity_arr = vtk.vtkDoubleArray()
        velocity_arr.SetNumberOfComponents(3)
        velocity_arr.SetName("Velocity")
        id_arr = vtk.vtkIntArray()
        id_arr.SetName("ID")

        for i in range(n_particles):
            if data_start + i >= len(lines):
                break
            parts = lines[data_start + i].split()
            if len(parts) < 10:
                continue
            pid = int(parts[0])
            rad = float(parts[2])
            px, py, pz = float(parts[4]), float(parts[5]), float(parts[6])
            vx, vy, vz = float(parts[7]), float(parts[8]), float(parts[9])

            points.InsertNextPoint(px, py, pz)
            radius_arr.InsertNextValue(rad)
            velocity_arr.InsertNextTuple3(vx, vy, vz)
            id_arr.InsertNextValue(pid)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().AddArray(radius_arr)
        polydata.GetPointData().AddArray(velocity_arr)
        polydata.GetPointData().AddArray(id_arr)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(os.path.join(output_dir, f"particle_{timestep_index:04d}.vtp"))
        writer.SetInputData(polydata)
        writer.Write()

        timestep_index += 1


# 使用
p4p_to_vtp_series(r"D:\CHen\LBM-DEM-3d\example\lbdem\single_particle_settle\Re31.9_fully\output.p4p", "particles_vtp")