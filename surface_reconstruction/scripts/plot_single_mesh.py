# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import open3d as o3d
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mesh_path', type=str, default='../log/surface_reconstruction/DiGS_surf_recon_experiment/result_meshes/gargoyle.ply',
                    help='path to mesh file for visualization')
args = parser.parse_args()

vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=800)

mesh = o3d.io.read_triangle_mesh(args.mesh_path)
mesh.compute_vertex_normals()
vis.add_geometry(mesh)
vis.run()
vis.destroy_window()

