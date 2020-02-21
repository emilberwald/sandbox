import functools
from typing import Iterable

import numpy as np
from pint import UnitRegistry

ur = UnitRegistry()


class TopologicalManifold:
    def __init__(self, atlas):
        self.atlas = atlas


class PseudoRiemannianManifold(TopologicalManifold):
    def __init__(self, atlas, co_metric: np.array):
        super().__init__(atlas)
        self.co_metric = co_metric

    def distance(self, contra_source: np.array, contra_target: np.array):
        return self.co_metric @ (contra_target - contra_source)


class SpaceTime(PseudoRiemannianManifold):
    def __init__(
        self, atlas, co_metric: np.array = np.array([[-ur.c ** 2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    ):
        super().__init__(atlas, co_metric)
        self.spatial_co_metric = np.array(
            [
                [
                    self.co_metric[i][j] - self.co_metric[0][i] * self.co_metric[0][j] / self.co_metric[0][0]
                    for j in range(1, 4)
                ]
                for i in range(1, 4)
            ]
        )

    def spatial_distance(self, contra_source: np.array, contra_target: np.array):
        return self.spatial_co_metric @ (contra_target - contra_source)


class Event:
    def __init__(self, spacetime: SpaceTime, time: float, spatial_position: np.array):
        self.spacetime = spacetime
        self.time = time
        self.position = spatial_position


class Particle(Event):
    def __init__(self, spacetime: SpaceTime, time: float, spatial_position: np.array, spatial_velocity: np.array):
        super().__init__(spacetime, time, spatial_position)
        self.velocity = spatial_velocity


class ElectricParticle(Particle):
    def __init__(
        self,
        spacetime: SpaceTime,
        time: float,
        spatial_position: np.array,
        spatial_velocity: np.array,
        electric_charge: float,
    ):
        super().__init__(spacetime, time, spatial_position, spatial_velocity)
        self.electric_charge = electric_charge


class RetardedElectromagneticWave:
    def __init__(self, source: ElectricParticle):
        self.source = source

    def shell_distance(self, contra_position: np.array, time: float):
        # rather than each point on the shell having its velocity,
        # we save the original position.
        # it is not very physical perhaps, it is easier to know which
        # way one is going rather than where one came from ...
        return time - np.linalg.norm(contra_position - self.source.position) / ur.c

    def scalar_potential(self, time: float):
        return (4 * np.pi * ur.eps_0) ** (-1) * self.source.electric_charge / (ur.c * (time - self.source.time))

    def vector_potential(self, time: float):
        return (
            ur.mu_0
            * (4 * np.pi) ** (-1)
            * self.source.electric_charge
            * self.source.velocity
            / (ur.c * (time - self.source.time))
        )


class OddTensorDensity:
    def __init__(self, co_metric: np.array, tensor_density_weight: float):
        detg = np.linalg.det(co_metric)
        self.factor = np.sign(detg) * np.abs(detg) ** (tensor_density_weight)


class LeviCevitaDensity(OddTensorDensity):
    def __init__(self, co_metric: np.array, tensor_density_weight: float):
        super().__init__(co_metric, tensor_density_weight)

    def __call__(self, *args):
        def ϵ(*args):
            return functools.reduce(
                lambda a, b: a * b,
                (np.sign(aj - ai) for i, ai in enumerate(args) for j, aj in enumerate(args) if i < j),
                1,
            )

        return self.factor * ϵ(*args)


class ElectroMagnetism:
    @staticmethod
    def electric_field(event: Event, retards, hit_distance: float):
        # we want to compute
        # E = - \nabla \phi - \partial_t A
        # https://en.wikipedia.org/wiki/Curvilinear_coordinates#Differentiation
        # ...


        dx = hit_distance
        dt = hit_distance / ur.c
        nabla_phi = np.array(0, 0, 0)
        dAt = np.array(0, 0, 0)
        for retard in retards:
            if retard.shell_distance(event.position, event.time - dt) < dx:
                dAt += retard.vector_potential(event.time - dt) / dt
            if retard.shell_distance(event.position, event.time) < dx:
                dAt += -retard.vector_potential(event.time) / dt

            # turn contra vector to covariant vector
            phi = retard.scalar_potential(event.time)

            # \partial_x A
            dxF = retard.shell_distance(event.position + np.array(dx, 0, 0), event.time) < dx
            dxB = retard.shell_distance(event.position - np.array(dx, 0, 0), event.time) < dx
            dphix = (dxF * phi - dxB * phi) / (2 * dx)
            # \partial_y A
            dyF = retard.shell_distance(event.position + np.array(0, dx, 0), event.time) < dx
            dyB = retard.shell_distance(event.position - np.array(0, dx, 0), event.time) < dx
            dphiy = (dyF * phi - dyB * phi) / (2 * dx)
            # \partial_z A
            dzF = retard.shell_distance(event.position + np.array(0, 0, dx), event.time) < dx
            dzB = retard.shell_distance(event.position - np.array(0, 0, dx), event.time) < dx
            dphiz = (dzF * phi - dzB * phi) / (2 * dx)

            nabla_phi += np.array([dphix, dphiy, dphiz])

        # to contra
        return np.linalg.inv(event.spacetime.spatial_co_metric) @ (nabla_phi - dAt)

    @staticmethod
    def magnetic_field(event: Event, retards: Iterable[RetardedElectromagneticWave], hit_distance: float):
        # we want to compute
        # B = curl(A)
        # https://en.wikipedia.org/wiki/Levi-Civita_symbol#Levi-Civita_tensors
        # https://en.wikipedia.org/wiki/Levi-Civita_symbol#Curl_(one_vector_field)
        # https://en.wikipedia.org/wiki/Curvilinear_coordinates#Differentiation
        # ...

        # contra version of levi cevita tensor density
        ε = LeviCevitaDensity(event.spacetime.spatial_co_metric, -0.5)

        curl_contra = np.array(0, 0, 0)
        for retard in retards:
            # turn contra vector to covariant vector
            A = event.spacetime.spatial_co_metric @ retard.vector_potential(event.time)

            # \partial_x A
            dxF = retard.shell_distance(event.position + np.array(hit_distance, 0, 0), event.time) < hit_distance
            dxB = retard.shell_distance(event.position - np.array(hit_distance, 0, 0), event.time) < hit_distance
            dAx = (dxF * A - dxB * A) / (2 * hit_distance)
            # \partial_y A
            dyF = retard.shell_distance(event.position + np.array(0, hit_distance, 0), event.time) < hit_distance
            dyB = retard.shell_distance(event.position - np.array(0, hit_distance, 0), event.time) < hit_distance
            dAy = (dyF * A - dyB * A) / (2 * hit_distance)
            # \partial_z A
            dzF = retard.shell_distance(event.position + np.array(0, 0, hit_distance), event.time) < hit_distance
            dzB = retard.shell_distance(event.position - np.array(0, 0, hit_distance), event.time) < hit_distance
            dAz = (dzF * A - dzB * A) / (2 * hit_distance)

            dA = [dAx, dAy, dAz]
            curl_contra += np.array(
                [
                    sum(ε(i, j, k) * dA[j][k] for j in range(0, curl_contra.size) for k in range(0, curl_contra.size))
                    for i in range(0, curl_contra.size)
                ]
            )
        return curl_contra

    @staticmethod
    def lorentz_force(retards: Iterable[RetardedElectromagneticWave], target: ElectricParticle, hit_distance: float):
        return target.electric_charge * (
            ElectroMagnetism.electric_field(target, retards, hit_distance)
            + np.cross(target.velocity, ElectroMagnetism.magnetic_field(target, retards, hit_distance))
        )


class Force:
    @staticmethod
    def delta(source, target):
        return target - source

    @staticmethod
    def distance(source, target):
        return np.linalg.norm(Force.delta(source, target))

    @staticmethod
    def gravity_on_target(source, target):
        return (
            -ur.G
            * source.mass
            * target.mass
            * Force.distance(source.position, target.position) ** (-3)
            * Force.delta(source.position, target.position)
        )

    @staticmethod
    def electromagnetism_on_target(
        historical_waves: Iterable[RetardedElectromagneticWave], target: ElectricParticle, hit_distance
    ):
        return ElectroMagnetism.lorentz_force(historical_waves, target, hit_distance)
