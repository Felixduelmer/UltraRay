import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt 

def unit_angle_z(v):
    temp = dr.asin(0.5 * dr.norm(mi.Vector3f(v.x, v.y, v.z - dr.mulsign(mi.Float(1.0), v.z)))) * 2
    return dr.select(v.z >= 0, temp, dr.pi - temp)


class ConvexTransducer(mi.Sensor):

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.radius = props.get('radius', 0.03)
        self.opening_angle = props.get('opening_angle', 60)
        self.elevational_height = props.get('elevational_height', 0.01)
        self.nb_elements_lat = props.get('nb_elements_lat', 128)
        self.pw_nb_angles = props.get('pw_nb_angles', 5)
        self.pw_start_angle = props.get('pw_start_angle', -10)
        self.pw_end_angle = props.get('pw_end_angle', 10)
        self.angles = dr.linspace(mi.Float, self.pw_start_angle, self.pw_end_angle, self.pw_nb_angles)
        dr.make_opaque(self.radius, self.opening_angle, self.nb_elements_lat, self.elevational_height, self.pw_start_angle, self.pw_nb_angles, self.pw_end_angle, self.angles)

    def calculate_pw_delays(self, ray_o, angle):
            # calculate the delays for the plane wave
            distances = ray_o.x * dr.sin(angle * dr.pi / 180)
            distances += ray_o.z
            distances -= dr.min(ray_o.z)
            distances -= dr.min(distances)

            return distances

    def sample_ray(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        # Sample wavelengths
        wavelengths, wav_weight = self.sample_wavelengths(
            dr.zeros(mi.SurfaceInteraction3f), wavelength_sample, active
        )

        # Define the local coordinate frame
        # TODO: this has to be based on the to_world transform
        self.frame = mi.Frame3f([0, 0, 1])

        # Sample direction in cylindrical coordinates
        phi = (position_sample.x - 0.5) * (self.opening_angle * dr.pi / 180)  # Convert to radians
        sin_phi, cos_phi = dr.sincos(phi)
        d_local = mi.Vector3f(sin_phi, 0, cos_phi)  # Direction in local space
        d_local_normalized = dr.normalize(d_local)

        # Sample origin in local coordinates
        y_offset = (position_sample.y - 0.5) * self.elevational_height  # Map y to [-height/2, height/2]
        o_local = mi.Point3f(0, y_offset, 0) + d_local_normalized * self.radius

        # use aperture sample for randorm direction in the aperture
        shoot_normal = False
        if shoot_normal:
            # # # # # Transform direction to world space
            d_world = self.frame.to_world(d_local_normalized)
            ray_delays = 0.0
        else:

            # map the x component of the aperture sample to a discrete angle
            angle_idx = dr.floor((aperture_sample.x * len(self.angles)))
            angle = dr.gather(mi.Float, self.angles, angle_idx)
            # compute the direction of the ray based on the angle in the xz plane
            sin_theta, cos_theta = dr.sincos(angle * dr.pi / 180)
            d_local_sample = mi.Vector3f(sin_theta, 0, cos_theta)

            # TODO: For this to be differentiable, we probably need to sample a bit here
            # d_local_sample = mi.warp.square_to_uniform_hemisphere(aperture_sample)
            # d_local_sample = mi.warp.square_to_cosine_hemisphere(aperture_sample)
            d_normalized = dr.normalize(d_local_sample)
            d_world = self.frame.to_world(d_normalized)
            # calculate the distance added to the ray
            # calculate small random delays for the plane wave
            ray_delays =  self.calculate_pw_delays(o_local, angle) 

        cos_weight = dr.sqr(dr.dot(d_local_normalized, d_normalized))

        # Transform origin to world space
        o_world = self.frame.to_world(o_local)

        return mi.Ray3f(o_world, d_world, time, wavelengths), wav_weight * cos_weight, angle_idx, ray_delays

    def sample_ray_differential(self, time, sample1, sample2, sample3, active=True):
        ray, weight, angle_idx, ray_delays  = self.sample_ray(time, sample1, sample2, sample3, active)
        return mi.RayDifferential3f(ray), weight, angle_idx, ray_delays 

    def sample_direction(self, it, sample, active=True):
        raise NotImplementedError("sample_direction not implemented")
    
    def surface_pos_to_uv(self, pos):

        loc_pos = self.frame.to_local(pos)
        phi = dr.atan2(loc_pos.x, loc_pos.z)
        # Map phi to [0, 1]
        half_opening_angle_rad = (self.opening_angle * dr.pi / 180) / 2
        phi = (phi + half_opening_angle_rad) / (2 * half_opening_angle_rad)

        resolution = self.film().crop_size()
    
        # Map y to [0, 1]
        y = (loc_pos.y + self.elevational_height / 2) / self.elevational_height
        point = mi.Point2f(phi, y) * resolution
        point = dr.floor(point)
        return point
    
        
    def aperture_contrib(self, center_idx, receiver_pos):
        return 1.0    

mi.register_sensor("convex_transducer", lambda props: ConvexTransducer(props))

if __name__ == "__main__":
    # visualize the sensor surface
    mi.set_variant("cuda_mono")
    import ultra_ray
    
    import matplotlib.pyplot as plt

    film = {
                'type': 'transducer_film',
                'height': 10,
                'width': 256,
                'max_distance': 0.2,
                'axial_resolution': 0.0001,
                'rfilter': {
                    'type': 'gaussian'
                },
            }
                # 'to_world': mi.ScalarTransform4f.scale([physical_width/2, (physical_height/2)* aspect_ratio, 0.1]),
            # },

    sensor : ConvexTransducer = mi.load_dict({
        'type' : 'convex_transducer',
        'radius': 0.03,
        'opening_angle': 60,
        'elevational_height': 0.01,
        'film' : film,
        'sampler': {
                    'type': 'stratified',
                    'sample_count': 1024,
                    'seed': 0,
                },
    })

    total = int(49)
    sampler = mi.load_dict({'type': 'orthogonal', 'seed': 0, 'sample_count': total})
    sampler.seed(seed=0, wavefront_size=total)

    samples = sampler.next_2d()
    ray, weight, angle_idx, ray_delays = sensor.sample_ray(0, sampler.next_1d(), sampler.next_2d(), sampler.next_2d())

    #visualize ray origin and direction
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ray.o.x, ray.o.y, ray.o.z)
    # ax.quiver(ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y, ray.d.z)
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([-0.1, 0.1])
    # add x, y,  z labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    