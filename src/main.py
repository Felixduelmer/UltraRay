
from contextlib import contextmanager

import mitsuba as mi
import drjit as dr
import time
mi.set_variant("cuda_mono")

from scene_loader import SceneLoader
from ultra_ray.beamformers.beamformer_ultraspy import beamform



def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    #scene arguments
    parser.add_argument("--scene", type=str, default='vertebrae',
                        help='scene file')
    
    # convex transducer arguments
    parser.add_argument("--td_radius_ax", type=float, default='0.03',
                        help='Radius of the transducer')
    parser.add_argument("--td_opening_angle", type=int, default='60',
                        help='Opening angle of the transducer')

    # general transducer arguments
    parser.add_argument("--td_elements_lat", type=int, default='128',
                        help='Number of elements in the lateral plane')
    parser.add_argument("--td_elements_ele", type=int, default='1',
                        help='Number of elements in the elevational plane')
    parser.add_argument("--td_ele_extent", type=float, default='0.005',
                        help='Elevational extent of the transducer')
    parser.add_argument("--fnumber", type=float, default='1.5',
                        help='F-number of the transducer')
    
    
    # plane wave imaging arguments
    parser.add_argument("--pw_nb_angles", type=int, default='5',
                        help='Number of angles to sample for plane wave imaging')
    parser.add_argument("--pw_start_angle", type=int, default='-10',
                        help='Start angle for plane wave imaging')
    parser.add_argument("--pw_end_angle", type=int, default='10',
                        help='End angle for plane wave imaging')
    

    parser.add_argument("--sampling_frequency", type=int, default='20000000',
                        help='Define the sampling frequency (used for beamforming)')
    parser.add_argument("--max_distance", type=float, default='0.2',
                        help='Max path length before being discarded')
    parser.add_argument("--central_frequency", type=int, default='5000000',
                        help='Central frequency of the chirp')
    parser.add_argument("--num_cycles", type=int, default='3',
                        help='Number of cycles of the chirp')
    
    # ray tracer arguments
    parser.add_argument("--sample_count", type=int, default='10000',
                        help='Number of samples per transducer element')
    parser.add_argument("--max_depth", type=int, default='10',
                        help='Maximum number of bounces')
    parser.add_argument("--temporal_filter", type=str, default='box',
                        help='Temporal filter to use in the axial direction either box or gaussian')
    parser.add_argument("--temporal_filter_std", type=float, default='2.0',
                        help='Standard deviation of the temporal filter (if gaussian)')

    # general arguments
    parser.add_argument("--SoS", type=float, default='1540',
                        help='Speed of sound')
    parser.add_argument("--attenuation_coefficient", type=float, default='0.2',
                        help='Attenuation coefficient for frequency dependent attenuation')
    return parser


@contextmanager
def time_annotator(label="Block"):
    start = time.time()
    yield
    end = time.time()
    print(f"Elapsed time for {label}: {end - start:.4f} seconds")



def simulate():
    # Parse scene arguments
    parser = config_parser()
    args = parser.parse_args()

    with time_annotator("scene setup"):
        scene_loader = SceneLoader(args)
        scene = scene_loader.define_scene(args.scene)
        # Setup beamforming as part of scene setup
        scene_loader.setup_beamforming()
        # Prepare integrator
        integrator = scene.integrator() 
        integrator.prepare_us_transducer(scene, sensor=0)

    with time_annotator("ray tracing"):
        rf_data = integrator.render(scene)
        dr.eval(rf_data)
        dr.sync_thread()

    with time_annotator("beamforming"):
        beamform(rf_data, args, visualize=False, save_image=True)
   
    
if __name__ == '__main__':

    simulate()