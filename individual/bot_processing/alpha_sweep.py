import os
import time
import numpy as np
import open3d as o3d

def load_stl(filepath):
    mesh = o3d.io.read_triangle_mesh(filepath)
    print(mesh.compute_vertex_normals())
    return mesh

def visualize_mesh(mesh):
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

def write_stl(mesh, filename):
    o3d.io.write_triangle_mesh(filename, mesh)

def alpha_shape_test(mesh, alphas, n_sample_points):
    SAVE_PATH = os.path.abspath(__file__ + "/../../") + "/binvox/"+ BOT_ID +"/alpha_test/"
    os.makedirs(SAVE_PATH, exist_ok=True)

    pcd = mesh.sample_points_poisson_disk(n_sample_points)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)

    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    for alpha in alphas:
        print(f"n_sample_points={n_sample_points}, alpha={alpha:.3f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha, tetra_mesh, pt_map)
        mesh.compute_vertex_normals()

        mesh.compute_triangle_normals()
        mesh.has_vertex_normals()
        print(mesh.compute_vertex_normals())

        write_stl(mesh, SAVE_PATH+f"body_alpha{alpha}_n{n_sample_points}.stl")

if __name__ == "__main__":
    global BOT_ID

    # N_SAMPLE_POINTS = [450000, 600000]
    # ALPHAS = [6, 12]

    BOT_ID = "Run4group0subject0"

    N_SAMPLE_POINTS = [450000]
    ALPHAS = [6, 8]
    

    # Load body mesh and compute alpha shape
    mesh = load_stl(os.path.abspath(__file__ + "/../../") + "/binvox/" + BOT_ID + "/body.stl")

    for n in N_SAMPLE_POINTS:
        start_time = time.time()
        alpha_shape_test(mesh, ALPHAS, n)
        print(f"{time.time()-start_time:.3f} seconds")
