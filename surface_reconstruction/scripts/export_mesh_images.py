# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import argparse
import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import trimesh

parser = argparse.ArgumentParser()
# parser.add_argument('--mesh_path', type=str, default='../log/surface_reconstruction/',
#                     help='path to top level directory containing different baseline results')
# parser.add_argument('--output_path', type=str, default='../log/surface_reconstruction/results/vis/',
#                     help='path to where output the restuls .txt file')
# parser.add_argument('--baseline_name', type=str, default='DiGS_surf_recon_experiment',
#                     help='name of baseline to generate visualizations for- i.e. subdirectory name')

# surface_reconstruction_digs_linear or surface_reconstruction_siren_wo_n_v2
# surface_reconstruction_sampling1 or surface_reconstruction_normals_1e3_v2
# /home/chamin/deep_geometric_prior_data/our_reconstructions
# /home/chamin/deep_geometric_prior_data/recon_pics
parser.add_argument('--mesh_path', type=str, default='log/surface_reconstruction_normals_1e3_v2/',
                    help='path to top level directory containing different baseline results')
parser.add_argument('--output_path', type=str, default='log/surface_reconstruction_digs_linear/results/vis/',
                    help='path to where output the restuls .txt file')
parser.add_argument('--baseline_name', type=str, default='DiGS_surf_recon_experiment',
                    help='name of baseline to generate visualizations for- i.e. subdirectory name')
args = parser.parse_args()

# python scripts/export_mesh_images.py --mesh_path log/surface_reconstruction_siren_wo_n_v2/ --output_path log/surface_reconstruction_siren_wo_n_v2/results/vis/

mesh_dir_path = os.path.join(args.mesh_path, args.baseline_name, 'result_meshes/')
# output_path = os.path.join(args.output_path, args.baseline_name)
# mesh_dir_path = '/home/chamin/deep_geometric_prior_data/scans'
# mesh_dir_path = '/home/chamin/deep_geometric_prior_data/ground_truth'
# mesh_dir_path = '/home/chamin/deep_geometric_prior_data/our_reconstructions'
output_path = '/home/chamin/CVPR2022/DiBS/surface_reconstruction/gt_ims_1e3_v2'
mesh_file_list = [f for f in os.listdir(mesh_dir_path) if os.path.isfile(os.path.join(mesh_dir_path, f))]
# mesh_file_list = [f for f in mesh_file_list if 'iter' not in f]
mesh_file_list = [f for f in mesh_file_list if 'iter' not in f or 'gar' in f]
# mesh_file_list = [f for f in mesh_file_list if 'anchor' in f]

gt_pcd_path = '/home/chamin/deep_geometric_prior_data/ground_truth'

mesh_file_list.sort()

os.makedirs(output_path, exist_ok=True)
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=800)

from scipy.spatial import cKDTree as KDTree

for mesh_filename in mesh_file_list:
    # import pdb; pdb.set_trace()
    shape = ''
    for name in ['anchor', 'daratech', 'dc', 'gargoyle', 'lord_quas']:
        print()
        if name in mesh_filename:
            shape = name
    if shape == '':
        continue
    print(mesh_filename, shape)
    mesh_path = os.path.join(mesh_dir_path, mesh_filename)
    gt_pcd = o3d.io.read_point_cloud(os.path.join(gt_pcd_path, shape+'.xyz'))
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    print(mesh, np.asarray(gt_pcd.points).shape)

    gt_kd_tree = KDTree(np.asarray(gt_pcd.points))
    mesh_vertices = np.asarray(mesh.vertices)
    distances, vertex_ids = gt_kd_tree.query(mesh_vertices, workers=4)
    # mesh.vertex_colors = None
    # vertex_colors = np.tile(np.array([1, 0.706, 0]), (len(mesh_vertices),1)) * distances.reshape(-1,1)
    frac = 1 - distances.clip(0,1)
    normal_vertex_colors = np.tile(np.array([0.5, 0.5, 0.5]), (len(mesh_vertices),1))
    error_vertex_colors = np.tile(np.array([1, 0.706, 0]), (len(mesh_vertices),1))
    normal_vertex_colors = np.tile(np.array([0, 0, 1]), (len(mesh_vertices),1))
    error_vertex_colors = np.tile(np.array([1, 0, 0]), (len(mesh_vertices),1))
    # import pdb; pdb.set_trace()
    vertex_colors = frac.reshape(-1,1) * normal_vertex_colors + (1-frac.reshape(-1,1)) * error_vertex_colors
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    # mesh.paint_uniform_color([1, 0.706, 0])

    vis.add_geometry(mesh)
    print('added geometry')
    # uncomment the following lines if you have recorded camera params for the visualization
    # params = o3d.io.read_pinhole_camera_parameters("../results/camera_params/" + mesh_filename.split('.')[0] + ".json")
    # ctr = vis.get_view_control()
    # ctr.convert_from_pinhole_camera_parameters(params)
    # vis.run()
    # print('run')

    # ctr = vis.get_view_control()
    # ctr.set_zoom(0.8)
    # # r = 1
    # # theta0 = 45 * np.pi / 180.
    # # phi0 = 45 * np.pi / 180.
    # # CamX = r * np.sin(theta0) * np.sin(phi0)
    # # CamY = r * np.cos(theta0)
    # # CamZ = r * np.sin(theta0) * np.cos(phi0)
    # # UpX = -np.cos(theta0) * np.sin(phi0)
    # # UpY = np.sin(theta0)
    # # UpZ = -np.cos(theta0) * np.cos(phi0)

    # # ctr.set_front((CamX, CamY, CamZ))
    # # ctr.set_up((UpX, UpY, UpZ))
    # # ctr.set_lookat((0, 0, 0))

    # # For anchor
    # ctr.set_front((1,-1,0.5))
    # ctr.set_up((0,0,1))
    # ctr.set_lookat((0, 0, 0))
    
    # # import pdb; pdb.set_trace()
    # shape = ''
    # for name in ['anchor', 'daratech', 'dc', 'gargoyle', 'lord_quas']:
    #     if name in mesh_filename:
    #         shape = name
    # if shape == '':
    #     continue
    params = o3d.io.read_pinhole_camera_parameters('scripts/camera_params/{}.json'.format(shape))
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(params)

    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    # vis.capture_screen_image(os.path.join(output_path, mesh_filename + ".png"))
    img = np.uint8(np.array(vis.capture_screen_float_buffer(do_render=True)) * 255)
    fig = plt.imshow(img)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(os.path.join(output_path, mesh_filename + ".png"), bbox_inches='tight', pad_inches = 0)
    print('captured screen')
    vis.clear_geometries()
vis.destroy_window()


