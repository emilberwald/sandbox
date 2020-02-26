import logging

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from sandbox.forces import *

logging.basicConfig(level=logging.INFO)

ur.setup_matplotlib()

if __name__ == "__main__":
    force = Force()
    particles: List[ChargedParticle] = list()
    spacetime = SpaceTime(atlas=None, covariant_metric=SpaceTime.minkowski_metric_positive_eigenvalue_space)
    nof_particles = 10
    time = 0 * ur.s
    dx = 0.1 * ur.m
    dt = 0.1 * dx / ur.c
    for i in range(0, nof_particles):
        particles.append(
            ChargedParticle(
                spacetime=spacetime,
                time=time,
                spatial_position=np.random.rand(3) * dx * 10,
                spatial_velocity=ur.m / ur.s * np.zeros(3),
                charges=(1e10 * ur.coulomb, 1e10 * ur.kg),
            )
        )

    figure = pyplot.figure()
    axes3D = Axes3D(figure)
    waves: List[RetardedWave] = list()
    for i in range(0, 100):
        time = i * dt
        logging.info(f"T={time.to_base_units()}")
        for particle_no, particle in enumerate(particles):
            waves.append(RetardedWave(particle))

        wave_hits = False
        # quick and dirty. probably very unstable and not convergent...
        updated_particles: List[ChargedParticle] = list()
        for particle_no, particle in enumerate(particles):
            F_G = force.gravity_on_target(waves, particle, dt * ur.c)
            F_EM = force.electromagnetism_on_target(waves, particle, dx)
            ΣF = F_G + F_EM
            Δx = dt * particle.velocity
            Δv = (
                dt * ΣF / particle.charge(ur.gram)
                if particle.charge(ur.gram)
                else np.zeros_like(particle.velocity) * ur.m / ur.s
            )
            if any(ΣF):
                logging.info(f"\t{particle_no}\t{particle}")
                logging.info(f"\tΣF={ΣF.to_base_units()}")
                logging.info(f"\t\tF_G={F_G.to_base_units()}")
                logging.info(f"\t\tF_EM={F_EM.to_base_units()}")
                logging.info(f"\tΔx={Δx.to_base_units()}")
                logging.info(f"\tΔv={Δv.to_base_units()}")
                wave_hits = True

            updated_particles.append(
                ChargedParticle(
                    spacetime=spacetime,
                    time=time,
                    spatial_position=particle.position + Δx,
                    spatial_velocity=particle.velocity + Δv,
                    charges=particle.charges,
                )
            )
        particles = updated_particles
        axes3D.scatter(
            pintify([particle.position[0] for particle in particles]),
            pintify([particle.position[1] for particle in particles]),
            pintify([particle.position[2] for particle in particles]),
        )
        if wave_hits:
            pyplot.savefig(f"{i}_{time.to_base_units()}.png", bbox_inches="tight")
