#include <math.h>
#include <iostream>

#include "Kokkos_Core.hpp"
#include <Kokkos_Random.hpp>

struct Config {
  int Npix;
  int NpixVelocity;
  int Npt;
  double dx;
  double vmin;
  double vmax;
};

KOKKOS_INLINE_FUNCTION
double normal(double mu, double sigma, auto& generator) {
  // Use Box-Muller method to generate a normal distribution
  double u = generator.drand(0, 1);
  double v = generator.drand(0, 1);
  return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v) * sigma + mu;
}

Kokkos::View<double***> hypercube(
    Kokkos::View<double**>& xp,
    Kokkos::View<double**>& vp,
    const double u[3],  // x-axis
    const double v[3],  // y-axis
    const double O[3],  // Origin
    const Config cfg 
) {

  Kokkos::View<double***> view("hypercube", cfg.Npix, cfg.Npix, cfg.NpixVelocity);
  const double w[3] = {v[1]*u[2] - v[2]*u[1],
                       v[2]*u[0] - v[0]*u[2],
                       v[0]*u[1] - v[1]*u[0]};
  // std::cout << "w: " << w[0] << ", " << w[1] << ", " << w[2] << std::endl;
  const double dv = cfg.vmax - cfg.vmin;
  double view_tot = 0.0;

  Kokkos::parallel_reduce("hypercube", cfg.Npt, KOKKOS_LAMBDA(const int i, double& view_tot) {
    // Map input pos to hypercube xy plane
    const double x = (
      (xp(i, 0) - O[0]) * u[0] +
      (xp(i, 1) - O[1]) * u[1] +
      (xp(i, 2) - O[2]) * u[2]
    ) / cfg.dx + 0.5;
    const double y = (
      (xp(i, 0) - O[0]) * v[0] +
      (xp(i, 1) - O[1]) * v[1] +
      (xp(i, 2) - O[2]) * v[2]
    ) / cfg.dx + 0.5;

    // Map input vel to hypercube l.o.s.
    double v_los = (
      (w[0]*vp(i, 0) + w[1]*vp(i, 1) + w[2]*vp(i, 2) - cfg.vmin) / dv
    );

    // Project spaxel onto hypercube
    // TODO: This is a very naive implementation
    int ix = (int) (x * cfg.Npix);
    int iy = (int) (y * cfg.Npix);
    int iz = (int) (v_los * cfg.NpixVelocity);

    if (ix >= 0 && ix < cfg.Npix && iy >= 0 && iy < cfg.Npix && iz >= 0 && iz < cfg.NpixVelocity) {
      view(ix, iy, iz) += 1.0;
      view_tot += 1.0;
    }
    // std::cout << "v_los = " << v_los << " " << cfg.NpixVelocity << std::endl;
    // std::cout << "i=" << i << ", ix=" << ix << ", iy=" << iy << ", iz=" << iz << std::endl;
  }, view_tot);

  std::cout << "view_tot: " << view_tot << " (" << view_tot / cfg.Npt << ")" << std::endl;
  return view;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);

  double n[3] = {1.0, 0.0, 0.0};

  // Normalize n vector
  {
    double nn = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    n[0] /= nn;
    n[1] /= nn;
    n[2] /= nn;
  }

  const int Npix = 256;
  const int NpixVelocity = 128;
  const int Npt = 10;
  {
    // Create point view
    Kokkos::View<double**> xp("xp", Npt, 3);
    Kokkos::View<double**> vp("vp", Npt, 3);
    
    // Initialize RNG
    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

    Kokkos::parallel_for("xp", Npt, KOKKOS_LAMBDA(const int i) {
      xp(i, 0) = 0.0;
      xp(i, 1) = 0.0;
      xp(i, 2) = 0.0;

      auto generator = random_pool.get_state();
      vp(i, 0) = normal(0.5, 0.1, generator);
      vp(i, 1) = normal(0.5, 0.1, generator);
      vp(i, 2) = normal(0.5, 0.1, generator);
      random_pool.free_state(generator);
    });

    // Create hypercube view
    double u[3] = {0.0, 1.0, 0.0};
    double v[3] = {0.0, 0.0, -1.0};
    double O[3] = {0.0, 0.0, 0.0};

    Config cfg;
    cfg.Npix = Npix;
    cfg.NpixVelocity = NpixVelocity;
    cfg.Npt = Npt;
    cfg.dx = 1.0;
    cfg.vmin = 0.0;
    cfg.vmax = 1.0;

    Kokkos::View<double***> cube = hypercube(xp, vp, u, v, O, cfg);
  }

  Kokkos::finalize();
}