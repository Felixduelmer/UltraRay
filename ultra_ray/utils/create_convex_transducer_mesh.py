import os
import numpy as np
import trimesh

def create_transducer_mesh(radius, opening_angle, elevational_height, mesh_name, resolution=50):
    """
    Create a 3D mesh of a convex transducer based on its properties.

    Args:
        radius (float): Radius of the transducer.
        opening_angle (float): Opening angle in degrees.
        elevational_height (float): Height of the transducer in elevation.
        resolution (int): Number of segments to use for mesh resolution.

    Returns:
        trimesh.Trimesh: A triangular mesh representation of the transducer.
    """
    # Convert opening angle to radians
    opening_angle_rad = np.radians(opening_angle)
    
    # Define angular range for the transducer
    phi = np.linspace(-opening_angle_rad / 2, opening_angle_rad / 2, resolution)
    
    # Define elevational range
    y = np.linspace(-elevational_height / 2, elevational_height / 2, resolution)
    
    # Generate points on the surface
    vertices = []
    for yi in y:
        for phii in phi:
            x = radius * np.sin(phii)
            z = radius * np.cos(phii)
            vertices.append([x, yi, z])
    
    vertices = np.array(vertices)
    
    # Create faces
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx1 = i * resolution + j
            idx2 = idx1 + 1
            idx3 = idx1 + resolution
            idx4 = idx3 + 1
            # Triangle 1
            faces.append([idx1, idx2, idx3])
            # Triangle 2
            faces.append([idx2, idx4, idx3])
    
    faces = np.array(faces)
    
    # Create a Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Visualize the mesh
    mesh.show()
    # create directory if it doesn't exist
    if not os.path.exists("ultra_ray/transducer_meshes/"):
        os.makedirs("ultra_ray/transducer_meshes/")
    mesh.export("ultra_ray/transducer_meshes/"+ mesh_name)

# Example usage
if __name__ == "__main__":
    radius = 0.03
    opening_angle = 60
    elevational_height = 0.01
    resolution = 50  # Adjust resolution for finer or coarser mesh

    create_transducer_mesh(radius, opening_angle, elevational_height, resolution)