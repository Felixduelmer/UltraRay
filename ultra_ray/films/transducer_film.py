import mitsuba as mi
import drjit as dr

from mitsuba import is_monochromatic
from ultra_ray.render.echo_block import EchoBlock


class TransducerFilm(mi.Film):
    r"""

    .. film-transducer_film:

    Transducer Film
    -------------------------------------------------

    A specialized film for ultrasound transducers that stores echo data based on time-of-flight.

    This film extends Mitsuba 3's Film class to handle ultrasound-specific data storage and processing.
    It accumulates echo samples in time bins and applies frequency-dependent attenuation.

    .. pluginparameters::

     * - max_distance
       - |float|
       - Maximum imaging depth in meters (default: 0.20)

     * - axial_resolution
       - |float|
       - Resolution along the axial direction in meters (default: 0.01)

     * - center_frequency
       - |float|
       - Center frequency of the transducer in Hz (default: 5e6)

     * - nb_events
       - |int|
       - Number of transmission events (default: 5)

     * - attenuation_coefficient
       - |float|
       - Tissue attenuation coefficient in dB/(MHz·cm) (default: 0.2 for general tissue, 0.5 for liver)

    See also, from `mi.Film <https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_films.html>`_:
    
    * `width` (integer)
    * `height` (integer)
    * `crop_width` (integer)
    * `crop_height` (integer)
    * `crop_offset_x` (integer)
    * `crop_offset_y` (integer)
    * `sample_border` (bool)
    * `rfilter` (rfilter)
    """

    def __init__(self, props):
        super().__init__(props)
        # NOTE: Also inherits properties from mi.Film (see documentation for this class above)
        self.max_distance = props.get("max_distance", mi.Float(0.20)) 
        self.axial_resolution = props.get("axial_resolution", mi.Float(0.01))
        self.center_freq = props.get("center_frequency", mi.Float(5e6))
        self.nb_events = props.get("nb_events", mi.UInt(5))
        self.attenuation_coefficient = props.get("attenuation_coefficient", mi.Float(0.2))
        assert self.max_distance > self.axial_resolution
        self.num_bins =  dr.floor( self.max_distance / self.axial_resolution) 

        dr.make_opaque(self.max_distance, self.axial_resolution, self.num_bins, self.nb_events, self.center_freq, self.attenuation_coefficient)


    def add_echo_data(self, spec, distance, ray_delays, receiver_pos, event_idx, wavelengths, active, emitter_pos, ray_weight):
        """
        Add a path's contribution to the film
        * spec: Spectrum / contribution of the path
        * distance: distance traveled by the path (opl)
        * wavelengths: for spectral rendering, wavelengths sampled
        * active: mask
        * receiver_pos: receiver position
        * ray_weight: weight of the ray given by the sensor
        """
        idd = (distance + ray_delays)  / self.axial_resolution
        # Frequency dependent attenuation - Power law (e.g. https://www.sciencedirect.com/science/article/pii/S1361841520302395)
        # Configurable attenuation coefficient (0.2 for general tissue, 0.5 for liver)
        attenuation = self.attenuation_coefficient * self.center_freq * 1e-6

        distance_in_cm = distance * 1e2
        spec = spec * dr.power(10, -attenuation * distance_in_cm / 10) 
        echo_pos = mi.Vector3f(event_idx, receiver_pos.x, idd)

        mask = (idd >= 0) & (idd < self.num_bins)
        self.echo.put(
            pos=echo_pos,
            wavelengths=wavelengths,
            value=spec * ray_weight,
            alpha=mi.Float(0.0),
            weight=mi.Float(0.0),
            active = active & mask,
        )


    def prepare_transducer(self, size, rfilter):
        """
        Called before the rendering starts (stuff related to time dependent rendering)
        This function also allocates the needed number of channels depending on the variant
        """
        channel_count = 3 if is_monochromatic else 5
        self.echo = EchoBlock(
            size=size, channel_count=channel_count, rfilter=rfilter
        )

    def traverse(self, callback):
        super().traverse(callback)
        callback.put_parameter(
            "max_distance", self.max_distance, mi.ParamFlags.NonDifferentiable
        )
        callback.put_parameter(
            "axial_resolution", self.axial_resolution, mi.ParamFlags.NonDifferentiable
        )

    def parameters_changed(self, keys):
        super().parameters_changed(keys)

    def to_string(self):
        string = "TransducerFilm[\n"
        string += f"  size = {self.size()},\n"
        string += f"  crop_size = {self.crop_size()},\n"
        string += f"  crop_offset = {self.crop_offset()},\n"
        string += f"  sample_border = {self.sample_border()},\n"
        string += f"  filter = {self.rfilter()},\n"
        string += f"  max_distance = {self.max_distance},\n"
        string += f"  axial_resolution = {self.axial_resolution},\n"
        string += f"  attenuation_coefficient = {self.attenuation_coefficient},\n"
        string += f"]"
        return string


mi.register_film("transducer_film", lambda props: TransducerFilm(props))
