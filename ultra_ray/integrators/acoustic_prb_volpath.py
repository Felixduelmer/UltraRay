from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ultra_ray.integrators.common import EchoADIntegrator, mis_weight


def index_spectrum(spec, idx):
    m = spec[0]
    if mi.is_rgb:
        m[dr.eq(idx, 1)] = spec[1]
        m[dr.eq(idx, 2)] = spec[2]
    return m


class AcousticPRBVolpathIntegrator(EchoADIntegrator):
    r"""
    .. _integrator-echo_prbvolpath:

    Echo Volumetric Path Replay Backpropagation (:monosp:`echo_prbvolpath`)
    ---------------------------------------------------------------------------------

    This class implements a volumetric Path Replay Backpropagation (PRB) integrator including the time dimension.

    It has the following properties:

    * Emitter sampling (a.k.a. next event estimation).
    * Russian Roulette stopping criterion.
    * No reparameterization. This means that the integrator cannot be used for shape optimization (it will return incorrect/biased gradients for geometric parameters like vertex positions.)
    * Detached sampling. This means that the properties of ideal specular objects (e.g., the IOR of a glass vase) cannot be optimized.

    For details on PRB and differentiable delta tracking, see the paper:
    "Path Replay Backpropagation: Differentiating Light Paths using
    Constant Memory and Linear Time" (Proceedings of SIGGRAPH'21)
    by Delio Vicini, Sébastien Speierer, and Wenzel Jakob

    Author: Miguel Crespo (miguel.crespo@epfl.ch)
    """

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get('max_depth', 10)
        self.rr_depth = props.get('rr_depth', 5)
        self.hide_emitters = props.get('hide_emitters', False)

        self.use_nee = False
        self.nee_handle_homogeneous = False
        self.handle_null_scattering = False
        self.is_prepared = False

    def prepare_scene(self, scene):
        if self.is_prepared:
            return
        self.is_prepared = True
        self.use_nee = True

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
               add_echo,
               **kwargs  # Absorbs unused arguments
               ) -> Tuple[mi.Spectrum,
                          mi.Bool, mi.Spectrum]:
        self.prepare_scene(scene)

        if mode == dr.ADMode.Forward:
            raise RuntimeError("PRBVolpathIntegrator doesn't support "
                               "forward-mode differentiation!")

        is_primal = mode == dr.ADMode.Primal

        ray = mi.Ray3f(ray)
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0 if is_primal else state_in) # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        throughput = mi.Spectrum(1)                   # Path throughput weight
        active = mi.Bool(active)
        distance = mi.Float(0.0)                      # Distance of the path

        si = dr.zeros(mi.SurfaceInteraction3f)
        needs_intersection = mi.Bool(True)
        last_scatter_event = dr.zeros(mi.Interaction3f)
        last_scatter_direction_pdf = mi.Float(1.0)

        depth = mi.UInt32(0)
        valid_ray = mi.Bool(False)
        specular_chain = mi.Bool(True)


        loop = mi.Loop(name=f"Path Replay Backpropagation ({mode.name})",
                    state=lambda: (sampler, active, depth, ray, si,
                                    throughput, L, needs_intersection,
                                    last_scatter_event, specular_chain,
                                    last_scatter_direction_pdf, valid_ray, distance))
        while loop(active):
            active &= dr.any(dr.neq(throughput, 0.0))

            #--------------------- Perform russian roulette --------------------

            q = dr.minimum(dr.max(throughput), 0.99)
            perform_rr = (depth > self.rr_depth)
            active &= (sampler.next_1d(active) < q) | ~perform_rr
            throughput[perform_rr] = throughput * dr.rcp(q)

            active_surface = active

            with dr.resume_grad(when=not is_primal):

                #--------------------- Surface Interactions --------------------

                intersect = active_surface & needs_intersection
                si[intersect] = scene.ray_intersect(ray, intersect)
                distance[intersect] += si.t

                # ----------------- Intersection with emitters -----------------

                ray_from_camera = active_surface & dr.eq(depth, 0)
                count_direct = ray_from_camera | specular_chain
                emitter = si.emitter(scene)
                active_e = active_surface & dr.neq(emitter, None) & ~(dr.eq(depth, 0) & self.hide_emitters)

                # Get the PDF of sampling this emitter using next event estimation
                ds = mi.DirectionSample3f(scene, si, last_scatter_event)
                if self.use_nee:
                    emitter_pdf = scene.pdf_emitter_direction(last_scatter_event, ds, active_e)
                else:
                    emitter_pdf = 0.0
                emitted = emitter.eval(si, active_e)
                contrib = dr.select(count_direct, throughput * emitted,
                                    throughput * mis_weight(last_scatter_direction_pdf, emitter_pdf) * emitted)
                L[active_e] += dr.detach(contrib if is_primal else -contrib)
                if not is_primal and dr.grad_enabled(contrib):
                    dr.backward(δL * contrib)

                add_echo(contrib, distance, si.p, ray.wavelengths, active_e)

                active_surface &= si.is_valid()
                ctx = mi.BSDFContext()
                bsdf = si.bsdf(ray)

                # ---------------------- Emitter sampling ----------------------

                if self.use_nee:
                    active_e_surface = active_surface & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth) & (depth + 1 < self.max_depth)
                    active_e = active_e_surface 

                    nee_sampler = sampler if is_primal else sampler.clone()
                    emitted, ds, ei_pos = self.sample_emitter(si, active_e_surface,
                        scene, sampler, active_e, mode=dr.ADMode.Primal)

                    # Query the BSDF for that emitter-sampled direction
                    bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx, si, si.to_local(ds.d), active_e_surface)
                    nee_weight = bsdf_val
                    nee_directional_pdf = dr.select(ds.delta, 0.0, bsdf_pdf)
                    contrib = throughput * nee_weight * mis_weight(ds.pdf, nee_directional_pdf) * emitted

                    L[active_e] += dr.detach(contrib if is_primal else -contrib)

                    if not is_primal:
                        self.sample_emitter(si, active_e_surface,
                            scene, nee_sampler, active_e, adj_emitted=contrib,
                            δL=δL, mode=mode)

                        if dr.grad_enabled(nee_weight) or dr.grad_enabled(emitted):
                            dr.backward(δL * contrib)

                    add_echo(contrib, distance + ds.dist, ei_pos, ray.wavelengths, active_e)

                # ------------------------ BSDF sampling -----------------------

                with dr.suspend_grad():
                    bs, bsdf_weight = bsdf.sample(ctx, si,
                                                sampler.next_1d(active_surface),
                                                sampler.next_2d(active_surface),
                                                active_surface)
                    active_surface &= bs.pdf > 0

                bsdf_eval = bsdf.eval(ctx, si, bs.wo, active_surface)

                if not is_primal and dr.grad_enabled(bsdf_eval):
                    Lo = bsdf_eval * dr.detach(dr.select(active_surface, L / dr.maximum(1e-8, bsdf_eval), 0.0))
                    if mode == dr.ADMode.Backward:
                        dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

                throughput[active_surface] *= bsdf_weight
                bsdf_ray = si.spawn_ray(si.to_world(bs.wo))
                ray[active_surface] = bsdf_ray


                needs_intersection |= active_surface
                non_null_bsdf = active_surface & ~mi.has_flag(bs.sampled_type, mi.BSDFFlags.Null)
                depth[non_null_bsdf] += 1
                
                last_scatter_event[non_null_bsdf] = si
                last_scatter_direction_pdf[non_null_bsdf] = bs.pdf

                valid_ray |= non_null_bsdf
                specular_chain |= non_null_bsdf & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
                specular_chain &= ~(active_surface & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Smooth))
                active &= active_surface

        return L if is_primal else δL, valid_ray, L

    def sample_emitter(self, si, active_surface, scene, sampler,
                       active, adj_emitted=None, δL=None, mode=None):
        is_primal = mode == dr.ADMode.Primal

        active = mi.Bool(active)

        ref_interaction = dr.zeros(mi.Interaction3f)
        ref_interaction[active_surface] = si

        position_sample = sampler.next_2d(active)
        ds, emitter_val = scene.sample_emitter_direction(ref_interaction,
                                                         position_sample,
                                                         False, active)

        shape_interaction_point = ds.p
        ds = dr.detach(ds)
        invalid = dr.eq(ds.pdf, 0.0)
        emitter_val[invalid] = 0.0
        active &= ~invalid

        ray = ref_interaction.spawn_ray_to(ds.p)
        max_dist = mi.Float(ray.maxt)
        total_dist = mi.Float(0.0)
        si = dr.zeros(mi.SurfaceInteraction3f)
        needs_intersection = mi.Bool(True)
        transmittance = mi.Spectrum(1.0)
        loop = mi.Loop(name=f"PRB Next Event Estimation ({mode.name})",
                       state=lambda: (sampler, active, ray, total_dist,
                                      needs_intersection, si, transmittance))
        while loop(active):
            remaining_dist = max_dist - total_dist
            ray.maxt = dr.detach(remaining_dist)
            active &= remaining_dist > 0.0

            # This ray will not intersect if it reached the end of the segment
            needs_intersection &= active
            si[needs_intersection] = scene.ray_intersect(ray, needs_intersection)
            needs_intersection &= False
            active_surface = active
            tr_multiplier = mi.Spectrum(1.0)


            # Handle interactions with surfaces
            active_surface &= si.is_valid()
            bsdf = si.bsdf(ray)
            bsdf_val = bsdf.eval_null_transmission(si, active_surface)
            tr_multiplier[active_surface] *= bsdf_val

            if not is_primal and dr.grad_enabled(tr_multiplier):
                active_adj = active_surface & (tr_multiplier > 0.0)
                dr.backward(tr_multiplier * dr.detach(dr.select(active_adj, δL * adj_emitted / tr_multiplier, 0.0)))

            transmittance *= dr.detach(tr_multiplier)

            # Update the ray with new origin & t parameter
            ray[active_surface] = dr.detach(si.spawn_ray(mi.Vector3f(ray.d)))
            ray.maxt = dr.detach(remaining_dist)
            needs_intersection |= active_surface

            # Continue tracing through scene if non-zero weights exist
            active &= active_surface & dr.any(dr.neq(transmittance, 0.0))
            total_dist[active] += si.t

        return emitter_val * transmittance, ds, shape_interaction_point

    def firstSurface(self, scene, ray_, active_):
        ray = mi.Ray3f(ray_)
        active = mi.Mask(active_)
        distance = mi.Float(0.0)

        loop = mi.Loop("FirstSurfaceLoop")
        loop.put(lambda: (active, ray, distance))
        loop.init()
        while loop(active):
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()
            distance[active] = distance + si.t
            bs = si.bsdf(mi.RayDifferential3f(ray))
            active &= mi.has_flag(bs.flags(), mi.BSDFFlags.Null)
            newOrigin = si.p + ray.d * mi.math.RayEpsilon * 2
            ray = mi.Ray3f(newOrigin, ray.d, ray.time, ray.wavelengths)
        return distance


mi.register_integrator("acoustic_prbvolpath",
                       lambda props: AcousticPRBVolpathIntegrator(props))
