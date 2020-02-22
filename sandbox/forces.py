import functools
from typing import Iterable, List

import numpy as np
from pint import Quantity, UnitRegistry, Unit

ur = UnitRegistry()


def pintify(args: List[Quantity]):
    args = [arg.to_base_units() for arg in args]
    assert len(set((arg.units for arg in args))) == 1
    return np.array([arg.magnitude for arg in args]) * args[0].units


class OddTensorDensity:
    def __init__(self, covariant_metric: np.array, tensor_density_weight: float):
        """
        __init__ https://en.wikipedia.org/wiki/Tensor_density
        :math:`sgn(det(g))|det(g)|^{w}`

        :param covariant_metric: Covarant metric :math:`(g_{ij})_p`
        :type covariant_metric: np.array
        :param tensor_density_weight: Tensor density weight :math:`w`
        :type tensor_density_weight: float
        """
        detg = np.linalg.det(covariant_metric)
        self.factor = np.sign(detg) * np.abs(detg) ** (tensor_density_weight)


class LeviCevitaDensity(OddTensorDensity):
    def __init__(self, covariant_metric: np.array, tensor_density_weight: float):
        super().__init__(
            covariant_metric=covariant_metric, tensor_density_weight=tensor_density_weight,
        )

    def __call__(self, *args):
        def ϵ(*vals):
            return functools.reduce(
                lambda a, b: a * b,
                (np.sign(aj - ai) for i, ai in enumerate(vals) for j, aj in enumerate(vals) if i < j),
                1,
            )

        return self.factor * ϵ(*args)


class TopologicalManifold:
    def __init__(self, atlas):
        self.atlas = atlas


class PseudoRiemannianManifold(TopologicalManifold):
    def __init__(self, atlas, covariant_metric: np.array):
        super().__init__(atlas=atlas)
        self.covariant_metric = covariant_metric

    def distance(self, contra_source: np.array, contra_target: np.array):
        return np.sqrt((contra_target - contra_source) @ self.covariant_metric @ (contra_target - contra_source))


class SpaceTime(PseudoRiemannianManifold):
    def __init__(self, atlas, covariant_metric: np.array):
        super().__init__(atlas=atlas, covariant_metric=covariant_metric)
        self.spatial_covariant_metric = np.array(
            [
                [
                    self.covariant_metric[i][j]
                    - self.covariant_metric[0][i] * self.covariant_metric[0][j] / self.covariant_metric[0][0]
                    for j in range(1, 4)
                ]
                for i in range(1, 4)
            ]
        )
        self.levi_cevita_spatial_contravariant_tensor_density = LeviCevitaDensity(
            covariant_metric=self.spatial_covariant_metric, tensor_density_weight=-0.5
        )

    def spatial_distance(self, contra_source: np.array, contra_target: np.array):
        return np.sqrt(
            np.abs((contra_target - contra_source) @ self.spatial_covariant_metric @ (contra_target - contra_source))
        )

    minkowski_metric_positive_eigenvalue_space = np.diag(np.array([-1, 1, 1, 1]))

    minkowski_metric_positive_eigenvalue_time = np.diag(np.array([1, -1, -1, -1]))


class Event:
    def __init__(self, *, spacetime: SpaceTime, time: float, spatial_position: np.array):
        self.spacetime = spacetime
        self.time = time
        self.position = spatial_position


class Particle(Event):
    def __init__(self, *, spacetime: SpaceTime, time: float, spatial_position: np.array, spatial_velocity: np.array):
        super().__init__(spacetime=spacetime, time=time, spatial_position=spatial_position)
        self.velocity = spatial_velocity


class ChargedParticle(Particle):
    def __init__(
        self,
        *,
        spacetime: SpaceTime,
        time: float,
        spatial_position: np.array,
        spatial_velocity: np.array,
        charges: Iterable[Quantity],
    ):
        super().__init__(
            spacetime=spacetime, time=time, spatial_position=spatial_position, spatial_velocity=spatial_velocity
        )
        self.charges = charges

    def charge(self, unit: Unit):
        for charge in self.charges:
            if charge.dimensionality == unit.dimensionality:
                return charge
        return 0 * unit


class RetardedWave:
    def __init__(self, source: ChargedParticle):
        self.source = source

    def shell_distance(self, contra_position: np.array, time: Quantity):
        # rather than each point on the shell having its velocity,
        # we save the original position.
        # it is not very physical perhaps, it is easier to know which
        # way one is going rather than where one came from ...
        distance = time * ur.c - self.source.spacetime.spatial_distance(self.source.position, contra_position)
        return distance

    def scalar_potential_electromagnetic(self, time: Quantity):
        if time == self.source.time:
            return 0 * ur.parse_expression("kilogram * meter ** 2 / ampere / second ** 3")
        return (
            (4 * np.pi * ur.eps_0) ** (-1)
            * self.source.charge(ur.elementary_charge)
            * (ur.c * (time - self.source.time)) ** (-1)
        )

    def vector_potential_electromagnetic(self, time: Quantity):
        if time == self.source.time:
            return np.zeros_like(self.source.velocity) * ur.parse_expression("kilogram * meter / ampere / second ** 2")
        return (
            ur.mu_0
            * (4 * np.pi) ** (-1)
            * self.source.charge(ur.elementary_charge)
            * self.source.velocity
            * (ur.c * (time - self.source.time)) ** (-1)
        )

    def scalar_potential_gravitational(self, time: Quantity):
        """
            - G m |dx| **(-3)
        :param time:
        :return:
        """
        if time == self.source.time:
            return 0 * ur.parse_expression("1 / second ** 2")
        return -1 * ur.gravitational_constant * self.source.charge(ur.gram) * (ur.c * (time - self.source.time)) ** (-3)


class Gravity:
    @staticmethod
    def scalar_potential_contribution(event: Event, retard_pot: RetardedWave, hit_distance: Quantity):
        hit = retard_pot.shell_distance(event.position, event.time) < hit_distance
        return hit * retard_pot.scalar_potential_gravitational(event.time)

    @staticmethod
    def gravitational_force(retard_pots: Iterable[RetardedWave], target: ChargedParticle, hit_distance: Quantity):
        # we want to compute - G m_1 m_2 |r2 - r1|**(-3) (r2 - r1)
        return sum(
            target.charge(ur.gram)
            * Gravity.scalar_potential_contribution(target, retard_pot, hit_distance)
            * (target.position - retard_pot.source.position)
            for retard_pot in retard_pots
        )


class ElectroMagnetism:
    @staticmethod
    def scalar_potential_gradient_contribution(event: Event, retard_pot: RetardedWave, hit_distance: Quantity):
        phi = retard_pot.scalar_potential_electromagnetic(event.time)
        # \partial_x \phi
        dxP = (
            retard_pot.shell_distance(
                event.position + np.array([hit_distance.magnitude, 0, 0]) * hit_distance.units, event.time
            )
            < hit_distance
        )
        dxM = (
            retard_pot.shell_distance(
                event.position - np.array([hit_distance.magnitude, 0, 0]) * hit_distance.units, event.time
            )
            < hit_distance
        )
        df0x = (dxP * phi - dxM * phi) / (2 * hit_distance)
        # \partial_y \phi
        dyP = (
            retard_pot.shell_distance(
                event.position + np.array([0, hit_distance.magnitude, 0]) * hit_distance.units, event.time
            )
            < hit_distance
        )
        dyM = (
            retard_pot.shell_distance(
                event.position - np.array([0, hit_distance.magnitude, 0]) * hit_distance.units, event.time
            )
            < hit_distance
        )
        df0y = (dyP * phi - dyM * phi) / (2 * hit_distance)
        # \partial_z \phi
        dzP = (
            retard_pot.shell_distance(
                event.position + np.array([0, 0, hit_distance.magnitude]) * hit_distance.units, event.time
            )
            < hit_distance
        )
        dzM = (
            retard_pot.shell_distance(
                event.position - np.array([0, 0, hit_distance.magnitude]) * hit_distance.units, event.time
            )
            < hit_distance
        )
        df0z = (dzP * phi - dzM * phi) / (2 * hit_distance)
        return pintify([df0x, df0y, df0z])

    @staticmethod
    def vector_potential_partial_time_derivative_contribution(
        event: Event, retard_pot: RetardedWave, hit_distance: Quantity
    ):
        dt = hit_distance / ur.c
        AM = retard_pot.shell_distance(event.position, event.time - dt) < hit_distance
        AP = retard_pot.shell_distance(event.position, event.time) < hit_distance
        return AM * (retard_pot.vector_potential_electromagnetic(event.time - dt) / dt) + AP * (
            -retard_pot.vector_potential_electromagnetic(event.time) / dt
        )

    @staticmethod
    def vector_potential_curl_contribution(event: Event, retard_pot: RetardedWave, hit_distance: Quantity):
        # turn contravariant vector to covariant vector
        A = event.spacetime.spatial_covariant_metric @ retard_pot.vector_potential_electromagnetic(event.time)

        # \partial_x A
        dxF = (
            retard_pot.shell_distance(
                event.position + np.array([hit_distance.magnitude, 0, 0]) * hit_distance.units, event.time
            )
            < hit_distance
        )
        dxB = (
            retard_pot.shell_distance(
                event.position - np.array([hit_distance.magnitude, 0, 0]) * hit_distance.units, event.time
            )
            < hit_distance
        )
        dAx = (dxF * A - dxB * A) / (2 * hit_distance)
        # \partial_y A
        dyF = (
            retard_pot.shell_distance(
                event.position + np.array([0, hit_distance.magnitude, 0]) * hit_distance.units, event.time
            )
            < hit_distance
        )
        dyB = (
            retard_pot.shell_distance(
                event.position - np.array([0, hit_distance.magnitude, 0]) * hit_distance.units, event.time
            )
            < hit_distance
        )
        dAy = (dyF * A - dyB * A) / (2 * hit_distance)
        # \partial_z A
        dzF = (
            retard_pot.shell_distance(
                event.position + np.array([0, 0, hit_distance.magnitude]) * hit_distance.units, event.time
            )
            < hit_distance
        )
        dzB = (
            retard_pot.shell_distance(
                event.position - np.array([0, 0, hit_distance.magnitude]) * hit_distance.units, event.time
            )
            < hit_distance
        )
        dAz = (dzF * A - dzB * A) / (2 * hit_distance)

        dA = [dAx, dAy, dAz]
        return pintify(
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
    def electric_field(
        event: Event, retard_pots: Iterable[RetardedWave], hit_distance: Quantity,
    ):
        # we want to compute
        # E = - \nabla \phi - \partial_t A
        # https://en.wikipedia.org/wiki/Curvilinear_coordinates#Differentiation
        # ...

        # covariant to contravariant
        return np.linalg.inv(event.spacetime.spatial_covariant_metric) @ (
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
    def magnetic_field(
        event: Event, retard_pots: Iterable[RetardedWave], hit_distance: Quantity,
    ):
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
        retard_pots: Iterable[RetardedWave], target: ChargedParticle, hit_distance: float,
    ):
        return target.charge(ur.elementary_charge) * (
            ElectroMagnetism.electric_field(target, retard_pots, hit_distance)
            + np.cross(target.velocity, ElectroMagnetism.magnetic_field(target, retard_pots, hit_distance),)
        )


class Force:
    @staticmethod
    def gravity_on_target(
        historical_waves: Iterable[RetardedWave], target: ChargedParticle, hit_distance: Quantity,
    ):
        return Gravity.gravitational_force(historical_waves, target, hit_distance)

    @staticmethod
    def electromagnetism_on_target(
        historical_waves: Iterable[RetardedWave], target: ChargedParticle, hit_distance: Quantity,
    ):
        return ElectroMagnetism.lorentz_force(historical_waves, target, hit_distance)
