# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import argparse
import os
import open3d as o3d
import numpy as np
import cv2



parser = argparse.ArgumentParser()
parser.add_argument('--mesh_path', type=str, default='../log/scene_reconstruction/DiGS_scene_recon_experiment/result_meshes/interior_room.ply',
                    help='path to top level directory containing different baseline results')
parser.add_argument('--output_path', type=str, default='../log/scene_reconstruction/', help='path to where output the restuls .txt file')
parser.add_argument('--baseline_name', type=str, default='DiGS_scene_recon_experiment',
                    help='name of baseline to generate visualizations for- i.e. subdirectory name')
args = parser.parse_args()

mesh_filename = args.mesh_path
output_path = os.path.join(args.output_path, args.baseline_name)

os.makedirs(output_path, exist_ok=True)
vis = o3d.visualization.Visualizer()
window_size = 1200
vis.create_window(width=window_size, height=window_size)

mesh = o3d.io.read_triangle_mesh(mesh_filename)
mesh.compute_vertex_normals()
vis.add_geometry(mesh)
ctr = vis.get_view_control()

res = 3
r = 1
n_frames = [50, 50, 150, 50]
theta_list = np.concatenate([np.linspace(65, 70, n_frames[0]), np.linspace(70, 150, n_frames[1]), 150*np.ones(n_frames[2]), 150*np.ones(n_frames[3])])
phi_list = np.concatenate([np.linspace(15, -18, n_frames[0]), -18*np.ones(n_frames[1]), np.linspace(-18, 360 - 18 + 90, n_frames[2]), 432*np.ones(n_frames[3])])
r_list = np.concatenate([np.linspace(1, 0.5, n_frames[0]), np.linspace(0.5, 0.2, n_frames[1]), 0.2*np.ones(n_frames[2]), 0.2*np.ones(n_frames[3])])
lookat = np.zeros([len(r_list), 3])
lookaty = np.concatenate([np.zeros(n_frames[0]), np.linspace(0, 1.1, n_frames[1]), 1.1*np.ones(n_frames[2]), 1.1*np.ones(n_frames[3])])
lookat[:, 1] = lookaty

out = cv2.VideoWriter( mesh_filename + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (window_size, window_size))
for i in range(len(theta_list)):
    ctr.set_lookat(lookat[i])
    theta0 = theta_list[i] * np.pi / 180.
    phi0 = phi_list[i] * np.pi / 180.
    ctr.set_zoom(r_list[i])
    CamX = r * np.sin(theta0) * np.sin(phi0) + lookat[i, 0]
    CamY = r * np.cos(theta0) + lookat[i, 1]
    CamZ = r * np.sin(theta0) * np.cos(phi0) + lookat[i, 2]
    UpX = -np.cos(theta0) * np.sin(phi0)
    UpY = np.sin(theta0)
    UpZ = -np.cos(theta0) * np.cos(phi0)
    ctr.set_front((CamX, CamY, CamZ))
    ctr.set_up((UpX, UpY, UpZ))
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    img = np.uint8(np.array(vis.capture_screen_float_buffer(do_render=True)) * 255)
    out.write(img)
out.release()
vis.clear_geometries()
vis.destroy_window()