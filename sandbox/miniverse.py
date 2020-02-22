import logging

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from sandbox.forces import *

logging.basicConfig(level=logging.DEBUG)

ur.setup_matplotlib()

if __name__ == "__main__":
    force = Force()
    particles: List[ChargedParticle] = list()
    spacetime = SpaceTime(atlas=None, covariant_metric=SpaceTime.minkowski_metric_positive_eigenvalue_time)
    time = 0 * ur.s
    dx = 0.1 * ur.m
    dt = 1 * ur.s
    for i in range(0, 10):
        particles.append(
            ChargedParticle(
                spacetime=spacetime,
                time=time,
                spatial_position=np.random.rand(3) * dx * 10,
                spatial_velocity=ur.m / ur.s * np.zeros(3),
                charges=(1 * ur.coulomb, 1 * ur.kg),
            )
        )

    waves: List[RetardedWave] = list()
    for i in range(0, 100):
        time = i * dt
        for particle_no, particle in enumerate(particles):
            waves.append(RetardedWave(particle))

        # quick and dirty. probably very unstable and not convergent...
        for particle_no, particle in enumerate(particles):
            F_G = force.gravity_on_target(waves, particle, dt * ur.c)
            F_EM = force.electromagnetism_on_target(waves, particle, dx)
            logging.info(f"F_G:{F_G}\tF_EM:{F_EM}")
            if particle.charge(ur.gram):
                particle.velocity += dt * (F_G + F_EM) / particle.charge(ur.gram)
            particle.position += dt * particle.velocity
        fig = pyplot.figure()
        ax = Axes3D(fig)
        ax.scatter(
            pintify([particle.position[0] for particle in particles]),
            pintify([particle.position[1] for particle in particles]),
            pintify([particle.position[2] for particle in particles]),
        )
        pyplot.savefig(f"{i}.png", bbox_inches="tight")
