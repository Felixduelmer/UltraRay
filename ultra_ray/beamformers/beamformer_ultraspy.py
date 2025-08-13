import ultraspy as us
import numpy as np
from ultraspy.scan import GridScan
from ultraspy.io.reader import Reader
from ultraspy.beamformers.das import DelayAndSum
from ultraspy.probes.convex_probe import ConvexProbe
from ultraspy.helpers.transmit_delays_helpers import compute_pw_delays
import matplotlib.pyplot as plt
import os
import time

# Try to import CuPy for GPU acceleration, fall back to NumPy for CPU
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    import numpy as cp  # Use NumPy as CuPy substitute
    HAS_CUPY = False

# Backend selection function
def get_backend():
    """Returns the appropriate backend (cupy or numpy)"""
    return cp if HAS_CUPY else np

def to_cpu(array):
    """Convert array to CPU (NumPy) format"""
    if HAS_CUPY and hasattr(array, 'get'):
        return array.get()  # CuPy to NumPy
    elif HAS_CUPY and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    else:
        return np.asarray(array)  # Already NumPy or convertible

def to_backend(array):
    """Convert array to current backend format"""
    if HAS_CUPY:
        return cp.asarray(array)
    else:
        return np.asarray(array)

def visualize_fn(b_mode, x, z, mask_outside, visualize, save_image):
    # set font size to 20
    plt.rcParams.update({'font.size': 20})
    
    # Convert everything to CPU for visualization
    b_mode_backend = to_backend(b_mode)
    mask_outside_backend = to_backend(mask_outside)
    b_mode_masked = get_backend().where(mask_outside_backend, 1, b_mode_backend)
    b_mode_cpu = to_cpu(b_mode_masked)
    
    # Convert coordinates to CPU
    x_cpu = to_cpu(x)
    z_cpu = to_cpu(z)
    
    # Pre-compute extent
    extent = [x_cpu[0] * 1e3, x_cpu[-1] * 1e3, z_cpu[-1] * 1e3, z_cpu[0] * 1e3]

    if visualize or save_image:
        plt.figure(figsize=(10, 8))
        plt.imshow(b_mode_cpu, extent=extent, clim=[-60, 0], cmap='gray')
        plt.colorbar()
        plt.title('PW Beamforming')
        plt.xlabel('Width (mm)')
        plt.ylabel('Depth (mm)')
        
        if save_image:
            # Save in results folder
            if not os.path.exists('results'):
                os.makedirs('results')
            plt.savefig(f'results/beamforming_{time.strftime("%Y_%m_%d_%H_%M_%S")}.png', dpi=300, bbox_inches='tight')
            print(f"Saved image to results/beamforming_{time.strftime('%Y_%m_%d_%H_%M_%S')}.png")
        
        if visualize:
            plt.show()
        else:
            plt.close() 

def compute_channel_delay(active_elements, nb_elements, lateral_extent, focal_point):
    backend = get_backend()
    ele_distance = backend.arange(-active_elements//2, active_elements//2)+1
    lat_distance = ele_distance * lateral_extent / (nb_elements)
    delays = backend.sqrt(backend.square(lat_distance) + backend.square(focal_point)) - focal_point
    return delays


class BeamformingSetup:
    """Class to store beamforming setup and configuration to avoid repeated initialization"""
    
    def __init__(self):
        self.beamformer = None
        self.scan = None
        self.x = None
        self.z = None
        self.mask_outside = None
        self.probe = None
        self.configured = False
    
    def setup_beamforming(self, args):
        """One-time setup of beamforming parameters and beamformer configuration"""
        
        backend = get_backend()
        
        # Pre-compute all static parameters
        nb_elements = args.td_elements_lat
        opening_angle = args.td_opening_angle
        radius = args.td_radius_ax
        opening_angle_rad = backend.radians(opening_angle)
        
        # Compute pitch
        pitch = radius * (opening_angle_rad / (nb_elements - 1))
        
        # Create probe (one-time setup)
        self.probe = ConvexProbe({
            'central_freq': args.central_frequency,
            'radius': radius,
            'pitch': float(to_cpu(pitch)),
            'nb_elements': nb_elements
        })
        
        # Pre-compute plane wave angles and delays
        angles = backend.linspace(args.pw_start_angle, args.pw_end_angle, args.pw_nb_angles)
        angles_rad = backend.radians(angles)
        angles_rad_cpu = to_cpu(angles_rad)
        delays = compute_pw_delays(angles_rad_cpu, self.probe, transmission_mode="positives", smallest=False)
        
        # Pre-compute sequence
        nb_transmissions = delays.shape[0]
        elements_indices = backend.arange(self.probe.nb_elements)
        sequence = {
            'emitted': to_cpu(backend.tile(elements_indices, (nb_transmissions, 1))),
            'received': to_cpu(backend.tile(elements_indices, (nb_transmissions, 1))),
        }
        
        # Setup acquisition info
        acquisition_info = {
            'sound_speed': args.SoS,
            't0': 0.0,
            'delays': delays,
            'sampling_freq': args.sampling_frequency,
            'prf': 1e-9,
            'signal_duration': 0,
            'sequence_elements': sequence
        }
        
        # Configure beamformer (one-time setup)
        self.beamformer = DelayAndSum(on_gpu=HAS_CUPY)
        self.beamformer.update_option('emitted_aperture', 'false')
        self.beamformer.update_option('rx_apodization', 'tukey')
        self.beamformer.update_option('interpolation', 'linear')
        self.beamformer.update_option('rx_apodization_alpha', 1.0)
        self.beamformer.update_option('reduction', 'sum')
        
        self.beamformer.update_setup('f_number', args.fnumber)
        self.beamformer.update_setup('bandwidth', 80.0)
        self.beamformer.automatic_setup(acquisition_info, self.probe)
        
        # Pre-compute grid and geometry (one-time setup)
        max_distance = args.max_distance/2
        wavelength = args.SoS / args.central_frequency
        division_factor = 10
        transducer_radius = args.td_radius_ax
        transducer_opening_angle = args.td_opening_angle
        half_angle = transducer_opening_angle / 2.0
        half_angle_rad = backend.radians(half_angle)
        
        # Batch compute trigonometric values
        cos_half_angle = backend.cos(half_angle_rad)
        sin_half_angle = backend.sin(half_angle_rad)
        
        z_offset = transducer_radius - transducer_radius * cos_half_angle
        max_lat_extent = (transducer_radius + max_distance) * sin_half_angle
        
        # Generate coordinate arrays
        step_size = wavelength / division_factor
        self.x = backend.arange(-max_lat_extent, max_lat_extent + wavelength/4, step_size)
        self.z = backend.arange(0, max_distance + z_offset + wavelength/4, step_size)
        
        # Pre-compute geometry limits and mask
        apex_limit = transducer_radius + max_distance
        apex_lower_limit = transducer_radius
        
        X, Z = backend.meshgrid(self.x, self.z, indexing='xy')
        Za = Z + transducer_radius * cos_half_angle
        
        dist_sq = X**2 + Za**2
        dist = backend.sqrt(dist_sq)
        angle_rad = backend.arctan2(X, Za)
        angle_deg = backend.degrees(angle_rad)
        
        self.mask_outside = (backend.abs(angle_deg) > half_angle) | (dist > apex_limit) | (dist < apex_lower_limit)
        
        # Setup scan grid
        x_cpu = to_cpu(self.x)
        z_cpu = to_cpu(self.z)
        self.scan = us.scan.GridScan(x_cpu, z_cpu)
        
        self.configured = True


# Global setup instance
_beamforming_setup = BeamformingSetup()


def setup_beamforming(args):
    """Call this once to initialize beamforming setup"""
    global _beamforming_setup
    if not _beamforming_setup.configured:
        _beamforming_setup.setup_beamforming(args)
    return _beamforming_setup


def beamform(data, args, visualize=False, save_image=False):
    
    # Ensure setup is complete
    global _beamforming_setup
    if not _beamforming_setup.configured:
        setup_beamforming(args)

    backend = get_backend()
    
    # Handle data conversion based on backend
    if HAS_CUPY:
        # Create a CUDA stream for optimized GPU operations
        stream = cp.cuda.Stream(non_blocking=True)
        
        with stream:
            if hasattr(data, '__dlpack__'):
                try:
                    # Use DLPack for zero-copy GPU-to-GPU transfer
                    dlpack_tensor = data.__dlpack__()
                    data = cp.from_dlpack(dlpack_tensor)
                except Exception as e:
                    print(f"DLPack conversion failed ({e}), falling back to standard conversion.")
                    data = cp.array(data)
            else:
                data = cp.array(data)
    else:
        # CPU mode - just ensure data is numpy
        data = np.asarray(data)

    # Use pre-configured beamformer and scan
    output = _beamforming_setup.beamformer.beamform(data, _beamforming_setup.scan)
    
    # Replace nans in output with 0
    output = backend.where(backend.isnan(output), 0, output)
    
    # Compute envelope using pre-configured beamformer
    envelope = _beamforming_setup.beamformer.compute_envelope(output, _beamforming_setup.scan)

    # Backend-based B-mode computation
    envelope_backend = to_backend(envelope)
    
    # Fused B-mode computation
    envelope_magnitude = backend.abs(envelope_backend)
    epsilon = backend.finfo(envelope_magnitude.dtype).tiny * 10
    b_mode_backend = 20 * backend.log10(backend.maximum(envelope_magnitude, epsilon))
    b_mode = b_mode_backend.T

    # Visualization using pre-computed coordinates and mask
    visualize_fn(b_mode, _beamforming_setup.x, _beamforming_setup.z, _beamforming_setup.mask_outside, visualize, save_image)

    return b_mode





def beamform_demo():
      # Load the data
    reader = Reader('data/resolution_distorsion_expe_dataset_rf.hdf5', 'picmus')
    print(reader)
    
    first_frame = reader.data[0]

    backend = get_backend()
    
    # Zone of interest
    x = backend.linspace(-20, 20, 500) * 1e-3
    z = backend.linspace(5, 50, 1000) * 1e-3
    # Convert arrays to NumPy for ultraspy compatibility
    x_cpu = to_cpu(x)
    z_cpu = to_cpu(z)
    scan = GridScan(x_cpu, z_cpu)

    # DAS Beamformer with appropriate backend
    beamformer = DelayAndSum(on_gpu=HAS_CUPY)

    # Set the information about the acquisition (probe, angles, ...) and the
    # options to use (here the transmit method, which needs to be centered for
    # PICMUS data).
    beamformer.automatic_setup(reader.acquisition_info, reader.probe)

    # Additional parameters
    beamformer.update_setup('f_number', 1.75)

    # Actual beamform, then compute the envelope of the beamformed signal
    d_output = beamformer.beamform(first_frame, scan)
    d_envelope = beamformer.compute_envelope(d_output, scan)

    # Get the B-Mode to display
    us.to_b_mode(d_envelope)
    b_mode = d_envelope.get()

    # Actual display
    import matplotlib.pyplot as plt
    # Convert arrays to NumPy for matplotlib compatibility
    x_cpu = to_cpu(x)
    z_cpu = to_cpu(z)
    b_mode_cpu = to_cpu(b_mode)
    
    extent = [x_cpu * 1e3 for x_cpu in [x_cpu[0], x_cpu[-1], z_cpu[-1], z_cpu[0]]]  # In mm
    plt.imshow(b_mode_cpu.T, extent=extent, cmap='gray', clim=[-30, 0])
    plt.title('DAS on PICMUS - 75 plane waves')
    plt.xlabel('Axial (mm)')
    plt.ylabel('Depth (mm)')
    plt.show()


if __name__ == '__main__':
    beamform_demo()

