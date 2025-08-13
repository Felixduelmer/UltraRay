# Delayed parsing of type annotations
from __future__ import annotations as __annotations__

import mitsuba as mi
import drjit as dr
import gc

from typing import Union, Any, Callable, Optional, Tuple

# from mitsuba.ad.integrators.common import ADIntegrator  # type: ignore

class ADIntegrator(mi.CppADIntegrator):
    """
    Abstract base class of numerous differentiable integrators in Mitsuba

    .. pluginparameters::

     * - max_depth
       - |int|
       - Specifies the longest path depth in the generated output image (where -1
         corresponds to :math:`\\infty`). A value of 1 will only render directly
         visible light sources. 2 will lead to single-bounce (direct-only)
         illumination, and so on. (Default: 6)
     * - rr_depth
       - |int|
       - Specifies the path depth, at which the implementation will begin to use
         the *russian roulette* path termination criterion. For example, if set to
         1, then path generation many randomly cease after encountering directly
         visible surfaces. (Default: 5)
    """

    def __init__(self, props = mi.Properties()):
        super().__init__(props)

        max_depth = props.get('max_depth', 6)
        if max_depth < 0 and max_depth != -1:
            raise Exception("\"max_depth\" must be set to -1 (infinite) or a value >= 0")

        # Map -1 (infinity) to 2^32-1 bounces
        self.max_depth = max_depth if max_depth != -1 else 0xffffffff

        self.rr_depth = props.get('rr_depth', 5)
        if self.rr_depth <= 0:
            raise Exception("\"rr_depth\" must be set to a value greater than zero!")

    def to_string(self):
        return f'{type(self).__name__}[max_depth = {self.max_depth},' \
               f' rr_depth = { self.rr_depth }]'

    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:

        if not develop:
            raise Exception("develop=True must be specified when "
                            "invoking AD integrators")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aov_names()
            )

            # Generate a set of rays starting at the sensor
            ray, weight, pos = self.sample_rays(scene, sensor, sampler)

            # Launch the Monte Carlo sampling process in primal mode
            L, valid, aovs, _ = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                δaovs=None,
                state_in=None,
                active=mi.Bool(True)
            )

            # Prepare an ImageBlock as specified by the film
            block = film.create_block()

            # Only use the coalescing feature when rendering enough samples
            block.set_coalesce(block.coalesce() and spp >= 4)

            # Accumulate into the image block
            ADIntegrator._splat_to_block(
                block, film, pos,
                value=L * weight,
                weight=1.0,
                alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                aovs=aovs,
                wavelengths=ray.wavelengths
            )

            # Explicitly delete any remaining unused variables
            del sampler, ray, weight, pos, L, valid
            gc.collect()

            # Perform the weight division and return an image tensor
            film.put_block(block)

            return film.develop()

    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, self.aov_names())

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            ray, weight, pos = self.sample_rays(scene, sensor, sampler)

            with dr.resume_grad():
                L, valid, aovs, _ = self.sample(
                    mode=dr.ADMode.Forward,
                    scene=scene,
                    sampler=sampler,
                    ray=ray,
                    active=mi.Bool(True)
                )

                block = film.create_block()
                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)

                ADIntegrator._splat_to_block(
                    block, film, pos,
                    value=L * weight,
                    weight=1,
                    alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                    aovs=aovs,
                    wavelengths=ray.wavelengths
                )

                # Perform the weight division
                film.put_block(block)
                result_img = film.develop()

                # Propagate the gradients to the image tensor
                dr.forward_to(result_img)

        return dr.grad(result_img)

    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, self.aov_names())

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            ray, weight, pos = self.sample_rays(scene, sensor, sampler)

            with dr.resume_grad():
                L, valid, aovs, _ = self.sample(
                    mode=dr.ADMode.Backward,
                    scene=scene,
                    sampler=sampler,
                    ray=ray,
                    active=mi.Bool(True)
                )

                # Prepare an ImageBlock as specified by the film
                block = film.create_block()

                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)

                # Accumulate into the image block
                ADIntegrator._splat_to_block(
                    block, film, pos,
                    value=L * weight,
                    weight=1,
                    alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                    aovs=aovs,
                    wavelengths=ray.wavelengths
                )

                film.put_block(block)

                del valid
                gc.collect()

                # This step launches a kernel
                dr.schedule(block.tensor())
                image = film.develop()

                # Differentiate sample splatting and weight division steps to
                # retrieve the adjoint radiance
                dr.set_grad(image, grad_in)
                dr.enqueue(dr.ADMode.Backward, image)
                dr.traverse(mi.Float, dr.ADMode.Backward)

            # We don't need any of the outputs here
            del ray, weight, pos, block, sampler
            gc.collect()

            # Run kernel representing side effects of the above
            dr.eval()

    def sample_rays(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float]:
        """
        Sample a 2D grid of primary rays for a given sensor

        Returns a tuple containing

        - the set of sampled rays
        - a ray weight (usually 1 if the sensor's response function is sampled
          perfectly)
        - the continuous 2D image-space positions associated with each ray
        """

        film = sensor.film()
        film_size = film.crop_size()
        rfilter = film.rfilter()
        border_size = rfilter.border_size()

        if film.sample_border():
            film_size += 2 * border_size

        spp = sampler.sample_count()

        # Compute discrete sample position
        idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)

        # Try to avoid a division by an unknown constant if we can help it
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        # Compute the position on the image plane
        pos = mi.Vector2i()
        pos.y = idx // film_size[0]
        pos.x = dr.fma(-film_size[0], pos.y, idx)

        if film.sample_border():
            pos -= border_size

        pos += mi.Vector2i(film.crop_offset())

        # Cast to floating point and add random offset
        pos_f = mi.Vector2f(pos) + sampler.next_2d()

        # Re-scale the position to [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
        offset = -mi.ScalarVector2f(film.crop_offset()) * scale
        pos_adjusted = dr.fma(pos_f, scale, offset)

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        with dr.resume_grad():
            ray, weight, angle_idx, ray_delays = sensor.sample_ray_differential(
                time=time,
                sample1=wavelength_sample,
                sample2=pos_adjusted,
                sample3=aperture_sample
            )

        # With box filter, ignore random offset to prevent numerical instabilities
        splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f
        
        return ray, weight, splatting_pos, angle_idx, ray_delays

    def prepare(self,
                sensor: mi.Sensor,
                seed: int = 0,
                spp: int = 0,
                aovs: list = []):
        """
        Given a sensor and a desired number of samples per pixel, this function
        computes the necessary number of Monte Carlo samples and then suitably
        seeds the sampler underlying the sensor.

        Returns the created sampler and the final number of samples per pixel
        (which may differ from the requested amount depending on the type of
        ``Sampler`` being used)

        Parameter ``sensor`` (``int``, ``mi.Sensor``):
            Specify a sensor to render the scene from a different viewpoint.

        Parameter ``seed` (``int``)
            This parameter controls the initialization of the random number
            generator during the primal rendering step. It is crucial that you
            specify different seeds (e.g., an increasing sequence) if subsequent
            calls should produce statistically independent images (e.g. to
            de-correlate gradient-based optimization steps).

        Parameter ``spp`` (``int``):
            Optional parameter to override the number of samples per pixel for the
            primal rendering step. The value provided within the original scene
            specification takes precedence if ``spp=0``.
        """

        film = sensor.film()
        sampler = sensor.sampler().clone()

        if spp != 0:
            sampler.set_sample_count(spp)

        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)

        film_size = film.crop_size()

        if film.sample_border():
            film_size += 2 * film.rfilter().border_size()

        wavefront_size = dr.prod(film_size) * spp

        if wavefront_size > 2**32:
            raise Exception(
                "The total number of Monte Carlo samples required by this "
                "rendering task (%i) exceeds 2^32 = 4294967296. Please use "
                "fewer samples per pixel or render using multiple passes."
                % wavefront_size)

        sampler.seed(seed, wavefront_size)
        film.prepare(aovs)

        return sampler, spp

    def _splat_to_block(block: mi.ImageBlock,
                       film: mi.Film,
                       pos: mi.Point2f,
                       value: mi.Spectrum,
                       weight: mi.Float,
                       alpha: mi.Float,
                       aovs: Sequence[mi.Float],
                       wavelengths: mi.Spectrum):
        '''Helper function to splat values to a imageblock'''
        if (dr.all(mi.has_flag(film.flags(), mi.FilmFlags.Special))):
            aovs = film.prepare_sample(value, wavelengths,
                                       block.channel_count(),
                                       weight=weight,
                                       alpha=alpha)
            block.put(pos, aovs)
            del aovs
        else:
            if mi.is_spectral:
                rgb = mi.spectrum_to_srgb(value, wavelengths)
            elif mi.is_monochromatic:
                rgb = mi.Color3f(value.x)
            else:
                rgb = value
            if mi.has_flag(film.flags(), mi.FilmFlags.Alpha):
                aovs = [rgb.x, rgb.y, rgb.z, alpha, weight] + aovs
            else:
                aovs = [rgb.x, rgb.y, rgb.z, weight] + aovs
            block.put(pos, aovs)


    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               depth: mi.UInt32,
               δL: Optional[mi.Spectrum],
               δaovs: Optional[mi.Spectrum],
               state_in: Any,
               active: mi.Bool) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float]]:
        """
        This function does the main work of differentiable rendering and
        remains unimplemented here. It is provided by subclasses of the
        ``RBIntegrator`` interface.

        In those concrete implementations, the function performs a Monte Carlo
        random walk, implementing a number of different behaviors depending on
        the ``mode`` argument. For example in primal mode (``mode ==
        drjit.ADMode.Primal``), it behaves like a normal rendering algorithm
        and estimates the radiance incident along ``ray``.

        In forward mode (``mode == drjit.ADMode.Forward``), it estimates the
        derivative of the incident radiance for a set of scene parameters being
        differentiated. (This requires that these parameters are attached to
        the AD graph and have gradients specified via ``dr.set_grad()``)

        In backward mode (``mode == drjit.ADMode.Backward``), it takes adjoint
        radiance ``δL`` and accumulates it into differentiable scene parameters.

        You are normally *not* expected to directly call this function. Instead,
        use ``mi.render()`` , which performs various necessary
        setup steps to correctly use the functionality provided here.

        The parameters of this function are as follows:

        Parameter ``mode`` (``drjit.ADMode``)
            Specifies whether the rendering algorithm should run in primal or
            forward/backward derivative propagation mode

        Parameter ``scene`` (``mi.Scene``):
            Reference to the scene being rendered in a differentiable manner.

        Parameter ``sampler`` (``mi.Sampler``):
            A pre-seeded sample generator

        Parameter ``depth`` (``mi.UInt32``):
            Path depth of `ray` (typically set to zero). This is mainly useful
            for forward/backward differentiable rendering phases that need to
            obtain an incident radiance estimate. In this case, they may
            recursively invoke ``sample(mode=dr.ADMode.Primal)`` with a nonzero
            depth.

        Parameter ``δL`` (``mi.Spectrum``):
            When back-propagating gradients (``mode == drjit.ADMode.Backward``)
            the ``δL`` parameter should specify the adjoint radiance associated
            with each ray. Otherwise, it must be set to ``None``.

        Parameter ``state_in`` (``Any``):
            The primal phase of ``sample()`` returns a state vector as part of
            its return value. The forward/backward differential phases expect
            that this state vector is provided to them via this argument. When
            invoked in primal mode, it should be set to ``None``.

        Parameter ``active`` (``mi.Bool``):
            This mask array can optionally be used to indicate that some of
            the rays are disabled.

        The function returns a tuple ``(spec, valid, state_out)`` where

        Output ``spec`` (``mi.Spectrum``):
            Specifies the estimated radiance and differential radiance in
            primal and forward mode, respectively.

        Output ``valid`` (``mi.Bool``):
            Indicates whether the rays intersected a surface, which can be used
            to compute an alpha channel.

        Output ``aovs`` (``List[mi.Float]``):
            Integrators may return one or more arbitrary output variables (AOVs).
            The implementation has to guarantee that the number of returned AOVs
            matches the length of self.aov_names().

        """

        raise Exception('RBIntegrator does not provide the sample() method. '
                        'It should be implemented by subclasses that '
                        'specialize the abstract RBIntegrator interface.')



class EchoADIntegrator(ADIntegrator):
    r"""
    .. _integrator-echoadintegrator:

    Echo AD Integrator
    -----------------------
    
    Abstract base class for echo integrators.
    """

    def __init__(self, props=mi.Properties()):
        super().__init__(props)

        # imported: max_depth and rr_depth

        # NOTE temporal_filter can take: box, gaussian, or chirp
        # (which sets it to use the same temporal filter same as the film's rfilter)
        self.temporal_filter = props.get("temporal_filter", "")
        self.gaussian_stddev = props.get("gaussian_stddev", 0.5)

    def to_string(self):
        return f"{type(self).__name__}[max_depth = {self.max_depth}, rr_depth = { self.rr_depth }]"
    
    def prepare(self, scene, sensor, seed, spp, aovs):
        film = sensor.film()
        original_sampler = sensor.sampler()
        sampler = original_sampler.clone()

        if spp != 0:
            sampler.set_sample_count(spp)

        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)

        film_size = film.crop_size()

        if film.sample_border():
            film_size += 2 * film.rfilter().border_size()

        wavefront_size = dr.prod(film_size) * spp
        # film.prepare(aovs)

        if wavefront_size <= 2**32:
            sampler.seed(seed, wavefront_size)
            # Intentionally pass it on a list to mantain compatibility
            return [(sampler, spp)]

        # It is not possible to render more than 2^32 samples
        # in a single pass (32-bit integer)
        # We reduce it even further, to 2^26, to make the progress
        # bar update more frequently at the cost of more overhead
        # to create the kernels/etc (measured ~5% overhead)
        spp_per_pass = int((2**26 - 1) / dr.prod(film_size))
        if spp_per_pass == 0:
            raise Exception("Your film is too big. Please make it smaller.")

        # Split into max-size jobs (maybe add reminder at the end)
        needs_remainder = spp % spp_per_pass != 0
        num_passes = spp // spp_per_pass + 1 * needs_remainder

        sampler.set_sample_count(num_passes)
        sampler.set_samples_per_wavefront(num_passes)
        sampler.seed(seed, num_passes)
        seeds = mi.UInt32(sampler.next_1d() * 2**32)

        def sampler_per_pass(i):
            if needs_remainder and i == num_passes - 1:
                spp_per_pass_i = spp % spp_per_pass
            else:
                spp_per_pass_i = spp_per_pass
            sampler_clone = sensor.sampler().clone()
            sampler_clone.set_sample_count(spp_per_pass_i)
            sampler_clone.set_samples_per_wavefront(spp_per_pass_i)
            sampler_clone.seed(seeds[i], dr.prod(film_size) * spp_per_pass_i)
            return sampler_clone, spp_per_pass_i

        return [sampler_per_pass(i) for i in range(num_passes)]

    def prepare_us_transducer(self, scene: mi.Scene, sensor: mi.Sensor):
        """
        Prepare the integrator to perform a transient simulation
        """
        import numpy as np

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        self.sensor = sensor
        film = sensor.film()
        from ultra_ray.films.transducer_film import TransducerFilm

        if not isinstance(film, TransducerFilm):
            raise AssertionError(
                "The film of the sensor must be of type transducer_film"
            )

        # Create the echo block responsible for storing the time contribution
        crop_size = film.crop_size()
        num_bins = film.num_bins

        # The size derives from the number of lateral transducer elements (for transmit and receive) and the number of bins
        size = np.array([film.nb_events, crop_size.x, num_bins])

        def load_filter(name, **kargs):
            """
            Shorthand for loading an specific reconstruction kernel
            """
            kargs["type"] = name
            f = mi.load_dict(kargs)
            return f

        def get_filters(sensor):
            """
            Selecting the temporal reconstruction filter.
            """
            if type(self.temporal_filter) == mi.ReconstructionFilter:
                time_filter = self.temporal_filter
            elif self.temporal_filter == "box":
                time_filter = load_filter("box")
            elif self.temporal_filter == "gaussian":
                stddev = max(self.gaussian_stddev, 0.5)
                time_filter = load_filter("gaussian", stddev=stddev)
            elif self.temporal_filter == "chirp":
                time_filter = load_filter("chirp")

            else:
                time_filter = sensor.film().rfilter()

            return [load_filter('box'), sensor.film().rfilter(), time_filter]
        filters = get_filters(sensor)
        film.prepare_transducer(size=size, rfilter=filters)
        self._film = film

    def add_echo_f(self, emitter_pos, center_idx, ray_delays, ray_weight, sample_scale):
        """
        Return a lambda function for saving echo samples.
        It pre-multiplies the sample scale.
        """
        return (
            lambda spec, distance, receiver_pos, wavelengths, active: self._film.add_echo_data(
                spec * sample_scale, distance, ray_delays, self.sensor.surface_pos_to_uv(receiver_pos), center_idx, wavelengths, active, emitter_pos, ray_weight * self.sensor.aperture_contrib(center_idx, receiver_pos)
            )
        )

    def render(
        self: mi.SamplingIntegrator,
        scene: mi.Scene,
        sensor: Union[int, mi.Sensor] = 0,
        seed: int = 0,
        spp: int = 0,
        develop: bool = True,
        evaluate: bool = True,
        progress_callback: function = None,
    ) -> mi.TensorXf:

        if not develop:
            raise Exception(
                "develop=True must be specified when " "invoking AD integrators"
            )

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            samplers_spps = self.prepare(
                sensor=sensor, scene = scene, seed=seed, spp=spp, aovs=self.aov_names()
            )

            total_spp = 0
            for _, spp_i in samplers_spps:
                total_spp += spp_i

            for i, (sampler, spp) in enumerate(samplers_spps):
                # Generate a set of rays starting at the sensor
                ray, ray_weight, emitter_pos, center_idx, ray_delays = self.sample_rays(scene, sensor, sampler)

                # Launch the Monte Carlo sampling process in primal mode
                L, valid, state = self.sample(
                    mode=dr.ADMode.Primal,
                    scene=scene,
                    sampler=sampler,
                    ray=ray,
                    depth=mi.UInt32(0),
                    δL=None,
                    state_in=None,
                    reparam=None,
                    active=mi.Bool(True),
                    max_distance=self._film.max_distance,
                    add_echo=self.add_echo_f(
                        emitter_pos=emitter_pos, center_idx = center_idx, ray_delays = ray_delays, ray_weight=ray_weight, sample_scale=1.0 / total_spp
                    ),
                )

                if progress_callback:
                    progress_callback((i + 1) / len(samplers_spps))

            # self.primal_image = film.steady.develop()
            echo_image = film.echo.develop()

            return echo_image

    def render_forward(
        self: mi.SamplingIntegrator,
        scene: mi.Scene,
        params: Any,
        sensor: Union[int, mi.Sensor] = 0,
        seed: int = 0,
        spp: int = 0,
    ) -> mi.TensorXf:
        # TODO implement render_forward (either here or move this function to RBIntegrator)
        raise NotImplementedError(
            "Check https://github.com/mitsuba-renderer/mitsuba3/blob/1e513ef94db0534f54a884f2aeab7204f6f1e3ed/src/python/python/ad/integrators/common.py"
        )

    def render_backward(
        self: mi.SamplingIntegrator,
        scene: mi.Scene,
        params: Any,
        grad_in: mi.TensorXf,
        sensor: Union[int, mi.Sensor] = 0,
        seed: int = 0,
        spp: int = 0,
    ) -> None:
        # TODO implement render_backward (either here or move this function to RBIntegrator)
        raise NotImplementedError(
            "Check https://github.com/mitsuba-renderer/mitsuba3/blob/1e513ef94db0534f54a884f2aeab7204f6f1e3ed/src/python/python/ad/integrators/common.py"
        )

    def sample(
        self,
        mode: dr.ADMode,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        depth: mi.UInt32,
        δL: Optional[mi.Spectrum],
        state_in: Any,
        reparam: Optional[
            Callable[[mi.Ray3f, mi.UInt32, mi.Bool],
                     Tuple[mi.Vector3f, mi.Float]]
        ],
        active: mi.Bool,
        add_echo,
        **kwargs  # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum, mi.Bool]:
        """
        This function does the main work of differentiable rendering and
        remains unimplemented here. It is provided by subclasses of the
        ADIntegrator interface.

        References:
        - https://github.com/diegoroyo/mitsuba3/blob/61c7cd1cff1937b2a041f1eacd90205b8e7e8c4a/src/python/python/ad/integrators/common.py#L489
        """

        raise Exception(
            "ADIntegrator does not provide the sample() method. "
            "It should be implemented by subclasses that "
            "specialize the abstract ADIntegrator interface."
        )


def mis_weight(pdf_a, pdf_b):
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    b2 = dr.sqr(pdf_b)
    w = a2 / (a2 + b2)
    return dr.detach(dr.select(dr.isfinite(w), w, 0))
