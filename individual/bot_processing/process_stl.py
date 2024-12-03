'''
Use open3d to..
 1. load in stl (http://www.open3d.org/docs/latest/tutorial/geometry/file_io.html?highlight=stl#Mesh)
 2. apply transformation (http://www.open3d.org/html/python_api/open3d.geometry.TriangleMesh.html#open3d.geometry.TriangleMesh.transform)
 3. compute alpha shape (http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html#Alpha-shapes)
 4. write out stl
'''

import os
import time
import numpy as np
import open3d as o3d

def load_stl(filepath):
    mesh = o3d.io.read_triangle_mesh(filepath)
    print(mesh.compute_vertex_normals())
    return mesh

def get_alpha_shape(mesh, n_sample_points=750, alpha=8):
    pcd = mesh.sample_points_poisson_disk(n_sample_points)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)

    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)

    mesh.compute_triangle_normals()
    mesh.has_vertex_normals()
    print(mesh.compute_vertex_normals())

    return mesh

def visualize_mesh(mesh):
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

def visualize_pcd(pcd):
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

def write_stl(mesh, filename):
    o3d.io.write_triangle_mesh(filename, mesh)

def load_transform_matrix(path):
    tform = np.zeros((4,4))
    with open(path, 'r') as f:
        # Read contents of file
        lines = f.readlines()
    ln_ct=0
    for line in lines:
        if line[0]!='#':
            nbrs = line.strip().split(' ')
            for i in range(len(nbrs)):
                tform[ln_ct,i] = nbrs[i]
            ln_ct+=1
    return tform

if __name__ == "__main__":

    start_time = time.time()

    BOT_ID = "Run8bot9"
    N_SAMPLE_POINTS = 450000
    ALPHA = 12
    
    # Load transform matrix
    TFORM_PATH = os.path.abspath(__file__ + "/../../") + "/binvox/"+BOT_ID+"/tform.mat"
    tform = load_transform_matrix(TFORM_PATH)
    
    # PROCESS BODY
    mesh = load_stl(os.path.abspath(__file__ + "/../../") + "/binvox/"+BOT_ID+"/body.stl")
    mesh = get_alpha_shape(mesh, n_sample_points=N_SAMPLE_POINTS, alpha=ALPHA)

    mesh.transform(tform)
    mesh_filename = os.path.abspath(__file__ + "/../../") + "/binvox/"+BOT_ID+"/"+f'body_tform_alpha{ALPHA}_pts{N_SAMPLE_POINTS}.stl'
    # mesh_filename = os.path.abspath(__file__ + "/../../") + "/binvox/"+BOT_ID+"/"+f'body_tform.stl'
    write_stl(mesh, mesh_filename)
    # os.system("binvox {}".format(mesh_filename))

    # # PROCESS CILIA
    cilia_mesh = load_stl(os.path.abspath(__file__ + "/../../") + "/binvox/"+BOT_ID+"/cilia.stl")
    cilia_mesh.transform(tform)
    cilia_filename = os.path.abspath(__file__ + "/../../") + "/binvox/"+BOT_ID+"/cilia_tform.stl"
    write_stl(cilia_mesh, cilia_filename)
    # # os.system("binvox {}".format(cilia_filename))


    print(f"{time.time()-start_time:.3f} seconds")