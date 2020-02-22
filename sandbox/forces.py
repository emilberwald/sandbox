import functools
from enum import Enum
from typing import Iterable

import numpy as np
from pint import Quantity, UnitRegistry

ur = UnitRegistry()


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


class TopologicalManifold:
    def __init__(self, atlas):
        self.atlas = atlas


class PseudoRiemannianManifold(TopologicalManifold):
    def __init__(self, co_metric: np.array, **kwargs):
        super().__init__(**kwargs)
        self.co_metric = co_metric

    def distance(self, contra_source: np.array, contra_target: np.array):
        return self.co_metric @ (contra_target - contra_source)


class SpaceTime(PseudoRiemannianManifold):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spatial_co_metric = np.array(
            [
                [
                    self.co_metric[i][j] - self.co_metric[0][i] * self.co_metric[0][j] / self.co_metric[0][0]
                    for j in range(1, 4)
                ]
                for i in range(1, 4)
            ]
        )
        self.levi_cevita_spatial_contravariant_tensor_density = LeviCevitaDensity(self.spatial_co_metric, -0.5)

    def spatial_distance(self, contra_source: np.array, contra_target: np.array):
        return self.spatial_co_metric @ (contra_target - contra_source)

    @property
    def minkowski_metric_positive_eigenvalue_space(self):
        return np.diag(np.array([-1, 1, 1, 1]))

    @property
    def minkowski_metric_positive_eigenvalue_time(self):
        return np.diag(np.array([1, -1, -1, -1]))


class Event:
    def __init__(self, spacetime: SpaceTime, time: float, spatial_position: np.array):
        self.spacetime = spacetime
        self.time = time
        self.position = spatial_position


class Particle(Event):
    def __init__(self, spatial_velocity: np.array, **kwargs):
        super().__init__(**kwargs)
        self.velocity = spatial_velocity


class ElectricParticle(Particle):
    def __init__(self, electric_charge: Quantity, **kwargs):
        super().__init__(**kwargs)
        self.electric_charge = electric_charge


class MassiveParticle(Particle):
    def __init__(self, mass: Quantity, **kwargs):
        super().__init__(**kwargs)
        self.mass = mass


class ElectricMassiveParticle(ElectricParticle, MassiveParticle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RetardedElectromagneticWave:
    def __init__(self, source: ElectricParticle):
        self.source = source

    def shell_distance(self, contra_position: np.array, time: Quantity):
        # rather than each point on the shell having its velocity,
        # we save the original position.
        # it is not very physical perhaps, it is easier to know which
        # way one is going rather than where one came from ...
        return time - self.source.spacetime.spatial_distance(self.source.position, contra_position) / ur.c

    def scalar_potential(self, time: Quantity):
        return (4 * np.pi * ur.eps_0) ** (-1) * self.source.electric_charge / (ur.c * (time - self.source.time))

    def vector_potential(self, time: Quantity):
        return (
            ur.mu_0
            * (4 * np.pi) ** (-1)
            * self.source.electric_charge
            * self.source.velocity
            / (ur.c * (time - self.source.time))
        )


class ElectroMagnetism:
    @staticmethod
    def scalar_potential_gradient_contribution(
        event: Event, retard_pot: RetardedElectromagneticWave, hit_distance: Quantity
    ):
        phi = retard_pot.scalar_potential(event.time)
        # \partial_x \phi
        dxP = retard_pot.shell_distance(event.position + np.array(hit_distance, 0, 0), event.time) < hit_distance
        dxM = retard_pot.shell_distance(event.position - np.array(hit_distance, 0, 0), event.time) < hit_distance
        df0x = (dxP * phi - dxM * phi) / (2 * hit_distance)
        # \partial_y \phi
        dyP = retard_pot.shell_distance(event.position + np.array(0, hit_distance, 0), event.time) < hit_distance
        dyM = retard_pot.shell_distance(event.position - np.array(0, hit_distance, 0), event.time) < hit_distance
        df0y = (dyP * phi - dyM * phi) / (2 * hit_distance)
        # \partial_z \phi
        dzP = retard_pot.shell_distance(event.position + np.array(0, 0, hit_distance), event.time) < hit_distance
        dzM = retard_pot.shell_distance(event.position - np.array(0, 0, hit_distance), event.time) < hit_distance
        df0z = (dzP * phi - dzM * phi) / (2 * hit_distance)
        return np.array([df0x, df0y, df0z])

    @staticmethod
    def vector_potential_partial_time_derivative_contribution(
        event: Event, retard_pot: RetardedElectromagneticWave, hit_distance: Quantity
    ):
        dt = hit_distance / ur.c
        dAt = np.array(0, 0, 0)
        if retard_pot.shell_distance(event.position, event.time - dt) < hit_distance:
            dAt += retard_pot.vector_potential(event.time - dt) / dt
        if retard_pot.shell_distance(event.position, event.time) < hit_distance:
            dAt += -retard_pot.vector_potential(event.time) / dt
        return dAt

    @staticmethod
    def vector_potential_curl_contribution(
        event: Event, retard_pot: RetardedElectromagneticWave, hit_distance: Quantity
    ):
        # turn contravariant vector to covariant vector
        A = event.spacetime.spatial_co_metric @ retard_pot.vector_potential(event.time)

        # \partial_x A
        dxF = retard_pot.shell_distance(event.position + np.array(hit_distance, 0, 0), event.time) < hit_distance
        dxB = retard_pot.shell_distance(event.position - np.array(hit_distance, 0, 0), event.time) < hit_distance
        dAx = (dxF * A - dxB * A) / (2 * hit_distance)
        # \partial_y A
        dyF = retard_pot.shell_distance(event.position + np.array(0, hit_distance, 0), event.time) < hit_distance
        dyB = retard_pot.shell_distance(event.position - np.array(0, hit_distance, 0), event.time) < hit_distance
        dAy = (dyF * A - dyB * A) / (2 * hit_distance)
        # \partial_z A
        dzF = retard_pot.shell_distance(event.position + np.array(0, 0, hit_distance), event.time) < hit_distance
        dzB = retard_pot.shell_distance(event.position - np.array(0, 0, hit_distance), event.time) < hit_distance
        dAz = (dzF * A - dzB * A) / (2 * hit_distance)

        dA = [dAx, dAy, dAz]
        return np.array(
            [
                sum(
                    event.spacetime.levi_cevita_spatial_contravariant_tensor_density(i, j, k) * dA[j][k]
                    for j in range(0, 3)
                    for k in range(0, 3)
                )
                for i in range(0, 3)
            ]
        )

    @staticmethod
    def electric_field(event: Event, retard_pots: Iterable[RetardedElectromagneticWave], hit_distance: Quantity):
        # we want to compute
        # E = - \nabla \phi - \partial_t A
        # https://en.wikipedia.org/wiki/Curvilinear_coordinates#Differentiation
        # ...

        # covariant to contravariant
        return np.linalg.inv(event.spacetime.spatial_co_metric) @ (
            -sum(
                ElectroMagnetism.scalar_potential_gradient_contribution(event, retard_pot, hit_distance)
                for retard_pot in retard_pots
            )
            - sum(
                ElectroMagnetism.vector_potential_partial_time_derivative_contribution(event, retard_pot, hit_distance)
                for retard_pot in retard_pots
            )
        )

    @staticmethod
    def magnetic_field(event: Event, retard_pots: Iterable[RetardedElectromagneticWave], hit_distance: Quantity):
        # we want to compute
        # B = curl(A)
        # https://en.wikipedia.org/wiki/Levi-Civita_symbol#Levi-Civita_tensors
        # https://en.wikipedia.org/wiki/Levi-Civita_symbol#Curl_(one_vector_field)
        # https://en.wikipedia.org/wiki/Curvilinear_coordinates#Differentiation
        # ...

        return sum(
            ElectroMagnetism.vector_potential_curl_contribution(event, retard_pot, hit_distance)
            for retard_pot in retard_pots
        )

    @staticmethod
    def lorentz_force(
        retard_pots: Iterable[RetardedElectromagneticWave], target: ElectricParticle, hit_distance: float
    ):
        return target.electric_charge * (
            ElectroMagnetism.electric_field(target, retard_pots, hit_distance)
            + np.cross(target.velocity, ElectroMagnetism.magnetic_field(target, retard_pots, hit_distance))
        )


class Force:
    @staticmethod
    def gravity_on_target(source: MassiveParticle, target):
        return (
            -ur.G
            * source.mass
            * target.mass
            * source.spacetime.spatial_distance(source.position, target.position) ** (-3)
            * (target.position - target.source)
        )

    @staticmethod
    def electromagnetism_on_target(
        historical_waves: Iterable[RetardedElectromagneticWave], target: ElectricParticle, hit_distance
    ):
        return ElectroMagnetism.lorentz_force(historical_waves, target, hit_distance)
