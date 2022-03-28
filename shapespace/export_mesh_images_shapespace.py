# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import argparse
import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
# parser.add_argument('--mesh_path', type=str, default='/mnt/IronWolf/logs/DiBS/3D/Baselines/surface_reconstruction/',
#                     help='path to top level directory containing different baseline results')
parser.add_argument('--mesh_path', type=str, default='/mnt/IronWolf/logs/DiBS/3D/grid_sampling_256/',
                    help='path to top level directory containing different baseline results')
parser.add_argument('--output_path', type=str,
                    default='/mnt/3.5TB_WD/PycharmProjects/DiBS/surface_reconstruction/results/vis/',
                    help='path to where output the restuls .txt file')
parser.add_argument('--baseline_name', type=str, default='DiBS',
                    help='name of baseline to generate visualizations for- i.e. subdirectory name')
args = parser.parse_args()

# mesh_dir_path = os.path.join(args.mesh_path, args.baseline_name, 'result_meshes/')
# output_path = os.path.join(args.output_path, args.baseline_name)
# mesh_file_list = [f for f in os.listdir(mesh_dir_path) if os.path.isfile(os.path.join(mesh_dir_path, f))]

mesh_dir_path = os.path.join('/home/chamin/PhD/DIBS/shapespace/DiBS/shapespace/' 'res_gt_latent')
output_path = os.path.join('/home/chamin/PhD/DIBS/shapespace/DiBS/shapespace/', 'res_gt_latent'+'_ims')
mesh_file_list = [f for f in os.listdir(mesh_dir_path) if os.path.isfile(os.path.join(mesh_dir_path, f))]

mesh_file_list = mesh_file_list[::-1]

os.makedirs(output_path, exist_ok=True)
o3d.visualization.RenderOption.mesh_show_back_face=True
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=800)

for mesh_filename in mesh_file_list:
    mesh_path = os.path.join(mesh_dir_path, mesh_filename)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    vis.add_geometry(mesh)
    # params = o3d.io.read_pinhole_camera_parameters("../results/camera_params/" + mesh_filename.split('.')[0] + ".json")
    params = o3d.io.read_pinhole_camera_parameters("params3.json")
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(params)
    vis.run()
    vis.capture_screen_image(os.path.join(output_path, mesh_filename + ".png"))
    vis.clear_geometries()
vis.destroy_window()
