# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import argparse
import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# # Stop tkinter error
# import matplotlib as mpl
# mpl.rcParams['agg.path.chunksize'] = 10000 # the default is 0

parser = argparse.ArgumentParser()
# parser.add_argument('--mesh_path', type=str, default='../log/surface_reconstruction/',
#                     help='path to top level directory containing different baseline results')
# parser.add_argument('--output_path', type=str, default='../log/surface_reconstruction/results/vis/',
#                     help='path to where output the restuls .txt file')
# parser.add_argument('--baseline_name', type=str, default='DiGS_surf_recon_experiment',
#                     help='name of baseline to generate visualizations for- i.e. subdirectory name')

# surface_reconstruction_digs_linear
# shapenet_recon_siren_wo_n
parser.add_argument('--mesh_path', type=str, default='log/surface_reconstruction_digs_linear/',
                    help='path to top level directory containing different baseline results')
parser.add_argument('--output_path', type=str, default='log/surface_reconstruction_digs_linear/results/vis/',
                    help='path to where output the restuls .txt file')
parser.add_argument('--baseline_name', type=str, default='DiGS_surf_recon_experiment',
                    help='name of baseline to generate visualizations for- i.e. subdirectory name')
args = parser.parse_args()

all_meshes_path = os.path.join(args.mesh_path, args.baseline_name)
all_output_path = os.path.join(args.output_path, args.baseline_name)
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=800)
for shape_class in os.listdir(all_meshes_path):
    print(shape_class)
    mesh_dir_path = os.path.join(all_meshes_path, shape_class, 'result_meshes/')
    output_path = all_output_path

    # mesh_dir_path = os.path.join(args.mesh_path, args.baseline_name, 'result_meshes/')
    # output_path = os.path.join(args.output_path, args.baseline_name)
    mesh_file_list = [f for f in os.listdir(mesh_dir_path) if os.path.isfile(os.path.join(mesh_dir_path, f))]
    mesh_file_list = [f for f in mesh_file_list if 'iter' not in f]

    mesh_file_list.sort()

    os.makedirs(output_path, exist_ok=True)

    for mesh_filename in mesh_file_list:
        mesh_path = os.path.join(mesh_dir_path, mesh_filename)
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        print(mesh)
        vis.add_geometry(mesh)
        print('\tadded geometry')
        # uncomment the following lines if you have recorded camera params for the visualization
        # params = o3d.io.read_pinhole_camera_parameters("../results/camera_params/" + mesh_filename.split('.')[0] + ".json")
        # ctr = vis.get_view_control()
        # ctr.convert_from_pinhole_camera_parameters(params)
        # vis.run()
        # print('run')

        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
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
        print('\tcaptured screen')
        vis.clear_geometries()
        del fig
        plt.close('all')
vis.destroy_window()


