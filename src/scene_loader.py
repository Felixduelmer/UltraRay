import mitsuba as mi
import numpy as np
from ultra_ray.beamformers.beamformer_ultraspy import setup_beamforming
from ultra_ray.utils.create_convex_transducer_mesh import create_transducer_mesh
import os


class SceneLoader():

    def __init__(self, args):
        self.args = args

        # transducer arguments
        self.td_transform = mi.ScalarTransform4f.translate([0, 0, 0])

        # convex transducer arguments
        self.td_radius_ax = args.td_radius_ax
        self.td_opening_angle = args.td_opening_angle
        
        # general transducer arguments
        self.td_ele_extent = args.td_ele_extent
        self.td_elements_lat = args.td_elements_lat
        self.td_elements_ele = args.td_elements_ele
        self.fnumber = args.fnumber
        self.td_ele_opening_angle = (180  / 360) * 2* np.pi
        
        # Setup transducer mesh first
        self.setup_transducer_mesh()
        self.td_mesh = args.transducer_mesh


        # plane wave imaging arguments
        self.pw_nb_angles = args.pw_nb_angles
        self.pw_start_angle = args.pw_start_angle
        self.pw_end_angle = args.pw_end_angle
        self.nb_events = args.pw_nb_angles

        # ray tracer arguments
        self.max_distance = args.max_distance
        self.max_depth = args.max_depth
        self.sample_count = args.sample_count
        self.sampling_frequency = args.sampling_frequency
        self.central_frequency = args.central_frequency
        self.num_cycles = args.num_cycles 
        self.SoS = args.SoS
        self.attenuation_coefficient = args.attenuation_coefficient
        self.axial_resolution = self.SoS / self.sampling_frequency

        # reconstruction arguments (PSF)
        self.temporal_filter = self.args.temporal_filter
        self.gaussian_stddev = self.args.temporal_filter_std
        
        # beamforming setup flag
        self.beamforming_setup_complete = False


    def load_cylinder(self, scene_dict):
        scene_dict['cylinder'] = {
            'type': 'obj',
            'filename': 'scenes/paper/cylinder/cylinder.obj',
            'flip_normals': False,
            'bsdf': {
                'type': 'diffuse'
            },
            'to_world': self.td_transform
        }
        return scene_dict


    def load_vertebrae(self, scene_dict):

        scene_dict['vertebrae'] = {
            'type': 'obj',
            'filename': 'scenes/paper/vertebrae/vertebrae.obj',
            'flip_normals': False,
            'bsdf': {
                'type': 'roughdielectric',
                'distribution': 'ggx',
                'alpha': 0.5,
                'int_ior': 7.8,
                'ext_ior': 1.54,
            },
            'to_world': self.td_transform
        }


        return scene_dict
        
    def define_scene(self, scene):
        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'acoustic_prbvolpath',
                'max_depth': self.max_depth,
                'temporal_filter':
                    {
                        'type': 'chirp',
                        'num_cycles': self.num_cycles,
                        'central_frequency': self.central_frequency,
                        'sampling_frequency': self.sampling_frequency,
                        'speed_of_sound': self.SoS,
                    }   
            },
        }
        film = { 'type': 'transducer_film',
                    'height': self.td_elements_ele,
                    'width': self.td_elements_lat,
                    'max_distance': self.max_distance,
                    'axial_resolution': self.axial_resolution,
                    'center_frequency': self.central_frequency,
                    'nb_events': self.nb_events,
                    'attenuation_coefficient': self.attenuation_coefficient,
                    'rfilter': {
                        'type': 'box',
                },
        }
        sampler = {
                'type': 'independent',
                'sample_count': self.sample_count
        }
        sensor = {
                'type': 'convex_transducer',
                'radius': self.td_radius_ax,
                'opening_angle': self.td_opening_angle,
                'elevational_height': self.td_ele_extent,
                'pw_nb_angles': self.pw_nb_angles,
                'pw_start_angle': self.pw_start_angle,
                'pw_end_angle': self.pw_end_angle,
                'film': film,
                'sampler': sampler,
                'to_world': self.td_transform,
                }
    
        scene_dict['sensor'] = sensor

        
        if scene == 'vertebrae':
            scene_dict = self.load_vertebrae(scene_dict)

        if scene == 'vertebrae_rotated':
            scene_dict = self.load_vertebrae(scene_dict)
            scene_dict['vertebrae']['to_world'] = mi.ScalarTransform4f.rotate([0, 0, 1], 90)

        if scene == "cylinder":
            scene_dict = self.load_cylinder(scene_dict)

        # Setup convex transducer as individual emitters
        phis = np.radians(np.linspace(-self.td_opening_angle/2, self.td_opening_angle/2, self.td_elements_lat) if self.td_elements_lat != 1  else np.array([0.0]))
        sin_phis = np.sin(phis)
        cos_phis = np.cos(phis)
        for idx, (sin_phi, cos_phi) in enumerate(zip(sin_phis, cos_phis)):
            scene_dict['emitter_' + str(idx)] = {
                'type': 'spot',
                'to_world': self.td_transform @ mi.ScalarTransform4f.look_at(
                    origin=[sin_phi*self.td_radius_ax, 0, cos_phi*self.td_radius_ax],
                    target=[sin_phi*2, 0, cos_phi*2],
                    up=[0, 1, 0]
                ),
                'intensity': 1.0,
                'cutoff_angle': 2.0,
                'beam_width': 2.0,
            }
        print("Setting up convex transducer")


        # Load the scene from the dictionary
        scene = mi.load_dict(scene_dict)

        return scene
    
    def setup_transducer_mesh(self):
        """Setup transducer mesh - create if it doesn't exist"""
        # Generate transducer mesh filename
        transducer_mesh = f"Convex_Transducer_{self.td_elements_lat}_{self.td_radius_ax}_{self.td_opening_angle}_{self.td_ele_extent}"
        # Replace all dots with underscores
        transducer_mesh = transducer_mesh.replace('.', '_')
        # Add .obj extension
        transducer_mesh += '.obj'
        
        # Check if mesh exists, create if needed
        transducer_meshes_dir = 'ultra_ray/transducer_meshes/'
        
        # Ensure the transducer_meshes directory exists
        if not os.path.exists(transducer_meshes_dir):
            print(f"Directory {transducer_meshes_dir} does not exist, creating it...")
            os.makedirs(transducer_meshes_dir, exist_ok=True)
        
        # Check if the specific mesh file exists
        try:
            existing_meshes = os.listdir(transducer_meshes_dir)
            mesh_exists = transducer_mesh in existing_meshes
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not access {transducer_meshes_dir}: {e}")
            mesh_exists = False
        
        if not mesh_exists:
            print(f"Probe mesh {transducer_mesh} not found in {transducer_meshes_dir}")
            print("Creating...")
            # Create the probe mesh
            create_transducer_mesh(self.args.td_radius_ax, self.args.td_opening_angle, self.args.td_ele_extent, transducer_mesh)
        
        # Store the mesh filename in args
        self.args.transducer_mesh = transducer_mesh
        print(f"Using transducer mesh: {transducer_mesh}")
    
    def setup_beamforming(self):
        """Setup beamforming configuration - call once after scene creation"""
        if not self.beamforming_setup_complete:
            setup_beamforming(self.args)
            self.beamforming_setup_complete = True
        else:
            print("Beamforming already configured - skipping setup") 


if __name__ == "__main__":
    mi.set_variant("cuda_mono")
    import ultra_ray
    SceneLoader().define_scene()