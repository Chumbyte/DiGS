# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import argparse
import os
import open3d as o3d
import numpy as np
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--mesh_path', type=str, default='./log/surface_reconstruction/',
                    help='path to top level directory containing different baseline results')
parser.add_argument('--output_path', type=str,
                    default='/mnt/3.5TB_WD/PycharmProjects/DiBS/surface_reconstruction/results/vid_vis/',
                    help='path to where output the restuls .txt file')
parser.add_argument('--gt_path', type=str,
                    default='/home/sitzikbs/Datasets/Reconstruction_Berger_Williams/GT/',
                    help='path togt files for camera view')
parser.add_argument('--baseline_name', type=str, default='DiGS_surf_recon_experiment',
                    help='name of baseline to generate visualizations for- i.e. subdirectory name')
args = parser.parse_args()

mesh_dir_path = os.path.join(args.mesh_path, args.baseline_name, 'result_meshes/')
output_path = os.path.join(args.output_path, args.baseline_name)
mesh_file_list = [f for f in os.listdir(mesh_dir_path) if os.path.isfile(os.path.join(mesh_dir_path, f))]

mesh_file_list.sort()

os.makedirs(output_path, exist_ok=True)
vis = o3d.visualization.Visualizer()



window_size = 800
for mesh_filename in mesh_file_list:

    vis.create_window(width=window_size, height=window_size)
    mesh_path = os.path.join(mesh_dir_path, mesh_filename)
    gt_mesh_path = os.path.join(args.gt_path, mesh_filename.replace('.ply', '.xyz'))
    out = cv2.VideoWriter(os.path.join(output_path, mesh_filename + '.webm'), cv2.VideoWriter_fourcc(*'VP90'), 30, (window_size, window_size))

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    gt_mesh = o3d.io.read_point_cloud(gt_mesh_path)
    if mesh_filename == 'anchor.ply' or mesh_filename == 'daratech.ply':
        mesh.rotate(mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0)))
        gt_mesh.rotate(mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0)))
    elif mesh_filename == 'gargoyle.ply':
        mesh.rotate(mesh.get_rotation_matrix_from_xyz((-np.pi, 0, 0)))
        gt_mesh.rotate(mesh.get_rotation_matrix_from_xyz((-np.pi, 0, 0)))
    mesh.compute_vertex_normals()


    vis.add_geometry(gt_mesh)  #load original mesh to get the right bounding box for view
    vis.clear_geometries()
    vis.add_geometry(mesh, reset_bounding_box=False)

    ctr = vis.get_view_control()

    ctr.set_zoom(0.8)

    res = 3
    r = 1

    theta0 = 45 * np.pi / 180.
    phi0 = 45 * np.pi / 180.
    CamX = r * np.sin(theta0) * np.sin(phi0)
    CamY = r * np.cos(theta0)
    CamZ = r * np.sin(theta0) * np.cos(phi0)
    UpX = -np.cos(theta0) * np.sin(phi0)
    UpY = np.sin(theta0)
    UpZ = -np.cos(theta0) * np.cos(phi0)

    ctr.set_front((CamX, CamY, CamZ))
    ctr.set_up((UpX, UpY, UpZ))
    ctr.set_lookat((0, 0, 0))

    for i in range(360//res + 1):
        theta = theta0
        phi = phi0 + i * res*np.pi/180
        CamX = r * np.sin(theta) * np.sin(phi)
        CamY = r * np.cos(theta)
        CamZ = r * np.sin(theta) * np.cos(phi)
        UpX = -np.cos(theta) * np.sin(phi)
        UpY = np.sin(theta)
        UpZ = -np.cos(theta) * np.cos(phi)

        ctr.set_front((CamX, CamY, CamZ))
        ctr.set_up((UpX, UpY, UpZ))
        ctr.set_lookat((0, 0, 0))

        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        img = np.uint8(np.array(vis.capture_screen_float_buffer(do_render=True)) * 255)
        out.write(img)
    out.release()

    vis.clear_geometries()


