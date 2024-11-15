#include <math.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>

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

#define SIGN(x) ((x) > 0 ? 1 : -1)

KOKKOS_INLINE_FUNCTION
double normal(double mu, double sigma, auto& generator) {
  // Use Box-Muller method to generate a normal distribution
  double u = generator.drand(0, 1);
  double v = generator.drand(0, 1);
  return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v) * sigma + mu;
}

// Define a point structure
struct Point {
    double x, y, z;

    KOKKOS_INLINE_FUNCTION
    constexpr Point operator+(const Point& p) const {
        return {x + p.x, y + p.y, z + p.z};
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Point operator+(const auto v) const {
        return {x + v, y + v, z + v};
    }


    KOKKOS_INLINE_FUNCTION
    constexpr Point operator-(const Point& p) const {
        return {x - p.x, y - p.y, z - p.z};
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Point operator*(const double& f) const {
        return {x * f, y * f, z * f};
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Point operator/(const double& f) const {
        return *this * (1/f);
    }

};

KOKKOS_INLINE_FUNCTION
constexpr double dot2d(const Point& a, const Point& b) {
  return a.x * b.x + a.y * b.y;
};

KOKKOS_INLINE_FUNCTION
constexpr double det2d(const Point& a, const Point& b) {
  return a.x * b.y - a.y * b.x;
}

KOKKOS_INLINE_FUNCTION
constexpr double dot3d(const Point& a, const Point& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
};

KOKKOS_INLINE_FUNCTION
bool check_in_parallelogram(const Point& PA, const Point& PQ, const Point& PR, const int signPQ, const int signPR, double out[2]) {
  double det_PQR = det2d(PQ, PR);

  if (det_PQR == 0) {
    out[0] = 0;
    out[1] = 0;
    return false;
  } else {
    out[0] = -det2d(PA, PQ) / det_PQR;
    out[1] = det2d(PA, PR) / det_PQR;

    if ((0 <= signPQ * out[0]) &
        (1 >= signPQ * out[0]) &
        (0 <= signPR * out[1]) &
        (1 >= signPR * out[1])
    ) {
      return true;
    } else {
      out[0] = -1;
      out[1] = -1;
      return false;
    }
  }
}

KOKKOS_INLINE_FUNCTION
void direct_integrate_cube(const Point& O, const Point& u, const Point& v, const Point& w, Kokkos::View<double***> buffer, Kokkos::View<bool**> buffer_mask) {
  const int Nx = buffer.extent(0);
  const int Ny = buffer.extent(1);

  const Point Oback = O + u + v + w;
  Point X({0, 0, 0});
  Point OfrontA({0, 0, 0}), ObackA({0, 0, 0});

  const double inv_dx = 1. / Nx;
  const double inv_dy = 1. / Ny;
  double nm[2] = {0, 0};

  auto minMax = [](
      const double O,
      const double Oback,
      const double u,
      const double v,
      const double w,
      const int N
  ) {
    const auto [xmin, xmax] = std::minmax(
      {O,     O + u,     O + v,     O + w,
       Oback, Oback - u, Oback - v, Oback - w}
    );
    return std::make_pair(
      int(floor((xmin > 0 ? xmin : 0) * N)),
      int(ceil((xmax < 1 ? xmax : 1) * N))
    );
  };

  auto [imin, imax] = minMax(O.x, Oback.x, u.x, v.x, w.x, Nx);
  auto [jmin, jmax] = minMax(O.y, Oback.y, u.y, v.y, w.y, Ny);

  for (auto i = imin; i < imax; ++i) {
    X.x = (i + 0.5) * inv_dx;

    OfrontA.x = X.x - O.x;
    ObackA.x = X.x - Oback.x;

    for (auto j = jmin; j < jmax; ++j) {
      auto zmin = std::numeric_limits<double>::infinity();
      auto zmax = -std::numeric_limits<double>::infinity();
      double z;
      uint Nhit = 0;
      X.y = (j + 0.5) * inv_dy;

      OfrontA.y = X.y - O.y;
      ObackA.y = X.y - Oback.y;

      bool within;
      within = check_in_parallelogram(OfrontA, v, u, 1, 1, nm);
      if (within) {
        z = O.z + nm[0] * u.z + nm[1] * v.z;
        zmin = fmin(z, zmin);
        zmax = fmax(z, zmax);
        Nhit++;
      }

      within = check_in_parallelogram(OfrontA, w, v, 1, 1, nm);
      if (within) {
        z = O.z + nm[0] * v.z + nm[1] * w.z;
        zmin = fmin(z, zmin);
        zmax = fmax(z, zmax);
        Nhit++;
      }

      within = check_in_parallelogram(OfrontA, w, u, 1, 1, nm);
      if (within) {
        z = O.z + nm[0] * u.z + nm[1] * w.z;
        zmin = fmin(z, zmin);
        zmax = fmax(z, zmax);
        Nhit++;
      }

      within = check_in_parallelogram(ObackA, v, u, -1, -1, nm);
      if (within) {
        z = Oback.z + nm[0] * u.z + nm[1] * v.z;
        zmin = fmin(z, zmin);
        zmax = fmax(z, zmax);
        Nhit++;
      }

      within = check_in_parallelogram(ObackA, w, v, -1, -1, nm);
      if (within) {
        z = Oback.z + nm[0] * v.z + nm[1] * w.z;
        zmin = fmin(z, zmin);
        zmax = fmax(z, zmax);
        Nhit++;
      }

      within = check_in_parallelogram(ObackA, w, u, -1, -1, nm);
      if (within) {
        z = Oback.z + nm[0] * u.z + nm[1] * w.z;
        zmin = fmin(z, zmin);
        zmax = fmax(z, zmax);
        Nhit++;
      }

      if (Nhit == 0) { // No hit
        continue;
      } else if (Nhit == 2) {
        buffer(i, j, 0) += zmax - zmin;
        buffer_mask(i, j) = 1;
      } else {
        // throw "This should not happen!";
      }
    }
  }
}


KOKKOS_INLINE_FUNCTION
double cross(const Point& a, const Point& b, const Point& c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
};

// Check if point (px, py) is inside the quadrangle defined by (p0, p1, p2, p3)
// using the cross product method.
KOKKOS_INLINE_FUNCTION
double locateOnQuad(const Point& p, const Point& u, const Point& v, const double dx) {
    // Calculate position of point p in (Ou)x(Ov) plane
    // We are in the face if tx and ty are in [-0.5, 0.5]
    const double tx = dot2d(p, u) / dx;
    const double ty = dot2d(p, v) / dx;

    // If all cross products have the same sign, the point is inside
    if      (tx <= 0) return 0;
    else if (tx >= 1) return 0;
    else if (ty <= 0) return 0;
    else if (ty >= 1) return 0;
    else              return tx * u.z * dx + ty * v.z * dx;
}


KOKKOS_INLINE_FUNCTION
void cell2hypercube(
  const Point& x, 
  const double dx,
  const Point& vel,
  const double sigma_v,
  const Point& u,
  const Point& v,
  const Point& w,
  const double weight,
  const int Npix,
  const int NpixVelocity,
  Kokkos::View<double***> buffer,
  Kokkos::View<bool**> mask
) {
  Point center = u * dot3d(x, u) + v * dot3d(x, v) + w * dot3d(x, w);
  Point p000 = center - u * dx/2 - v * dx/2 - w * dx/2;
  Point p111 = center + u * dx/2 + v * dx/2 + w * dx/2;

  direct_integrate_cube(p000, u * dx, v * dx, w * dx, buffer, mask);

  // // Define the 8 corners of the cell
  // const double dx_half_diag = (abs(u.x) + abs(v.x) + abs(w.x)) * dx / 2;
  // int imin = floor((0.5 + center.x - dx_half_diag) * Npix);
  // int imax = ceil((0.5 + center.x + dx_half_diag) * Npix);

  // const double dy_half_diag = (abs(u.y) + abs(v.y) + abs(w.y)) * dx / 2;
  // int jmin = floor((0.5 + center.y - dy_half_diag) * Npix);
  // int jmax = ceil((0.5 + center.y + dy_half_diag) * Npix);

  // const int iz = (int) (dot3d(vel, w) * NpixVelocity);

  // if (iz < 0 || iz >= NpixVelocity) {
  //   return;
  // }

  // // imin = std::max(0, imin);
  // // imax = std::min(imax, Npix-1);
  // // jmin = std::max(0, jmin);
  // // jmax = std::min(jmax, Npix-1);
  // // Point p {0, 0, 0};

  // // Draw a line from point A to point B
  // auto drawLine = [](const Point& A, const Point& B) {

  // };

  // auto interpolate = [&](const Point& u, const Point& v, const Point& O, const double dz) {
  //   // Start from origin, then pave (Ou, Ov)
  //   for (auto i = 0; i < 10; ++i) {
  //     for (auto j = 0; j < 10; ++j) {
  //       p = O + u * (i + 0.5) / 10 + v * (j + 0.5) / 10;
  //       int ix = int(p.x * Npix);
  //       int iy = int(p.y * Npix);

  //       if (ix < 0 || ix >= Npix || iy < 0 || iy >= Npix) {
  //         continue;
  //       }
  //       std::cout << "ix=" << ix << ", iy=" << iy << " p.z=" << p.z << std::endl;
  //       buffer(ix, iy, iz) += (p.z * weight);
  //     }
  //   }
  // };

  // const double dz = p111.z - p000.z;
  // p000.z = 0;
  // p111.z = 0;
  // interpolate(u, v, p000, 0);
  // interpolate(u, w, p000, 0);
  // interpolate(v, w, p000, 0);

  // interpolate(u*-1, v*-1, p111, dz);
  // interpolate(u*-1, w*-1, p111, dz);
  // interpolate(v*-1, w*-1, p111, dz);
  // p111.z += p000
  // for (auto i = imin; i <= imax; i++) {
  //   p.x = (i + 0.5) / Npix - 0.5;

  //   for (auto j = jmin; j <= jmax; j++) {
  //     p.y = (j + 0.5) / Npix - 0.5;

  //     // Check front-facing faces
  //     double b = 0;
  //     b += locateOnQuad(p - p000, u, v, dx);
  //     // b += locateOnQuad(p - p000, u, w, dx);
  //     // b += locateOnQuad(p - p000, v, w, dx);

  //     // // Check back-facing faces
  //     // b += locateOnQuad(p - p111, u, v, -dx);
  //     // b += locateOnQuad(p - p111, u, w, -dx);
  //     // b += locateOnQuad(p - p111, v, w, -dx);

  //     // Early exit does not intersect any of the cells
  //     if (b == 0) continue;

  //     b = abs(b);

  //     // Update the buffer
  //     Kokkos::atomic_add(&buffer(i, j, iz), b * weight);
  //   }
  // }
}

Kokkos::View<double***> hypercube(
    Kokkos::View<double**>& xc,
    Kokkos::View<double**>& vc,
    Kokkos::View<double*>& dxc,
    Kokkos::View<double*>& sigma_v,
    const Point u,  // x-axis
    const Point v,  // y-axis
    const Point O,  // Origin
    const Config cfg 
) {

  Kokkos::View<double***> view("hypercube", cfg.Npix, cfg.Npix, cfg.NpixVelocity);
  const Point w = {u.y*v.z - u.z*v.y,
                   u.z*v.x - u.x*v.z,
                   u.x*v.y - u.y*v.x};

  const double dv = cfg.vmax - cfg.vmin;
  double view_tot = 0.0;

  Kokkos::View<bool**> mask("mask", view.extent(0), view.extent(1));


  Kokkos::parallel_for("hypercube", cfg.Npt, KOKKOS_LAMBDA(const int i) {
    Point xcell = {
      (xc(i, 0) - O.x) / cfg.dx,
      (xc(i, 1) - O.y) / cfg.dx,
      (xc(i, 2) - O.z) / cfg.dx
    };
    Point vcell = {vc(i, 0), vc(i, 1), vc(i, 2)};

    // std::cout << i << std::endl;
    
    cell2hypercube(xcell, dxc(i), vcell, sigma_v(i), u, v, w, 1.0, cfg.Npix, cfg.NpixVelocity, view, mask);
  });

  Kokkos::parallel_reduce("sum", cfg.Npix, KOKKOS_LAMBDA(const int i, double& view_tot) {
    for (auto j = 0; j < view.extent(1); j++) {
      view_tot += view(i, j, 0);
    }
  }, view_tot);

  std::cout << "view_tot: " << view_tot << " (" << view_tot / cfg.Npt << ")" << std::endl;
  return view;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);

  const int Npix = 512;
  const int NpixVelocity = 32;
  const int Npt = 16 * 16;
  const int Npthalf = sqrt(Npt);
  {
    // Create point view
    Kokkos::View<double**> xc("xc", Npt, 3);
    Kokkos::View<double**> vc("vc", Npt, 3);
    Kokkos::View<double*> dxc("dxc", Npt);
    Kokkos::View<double*> sigma_vc("sigma_v", Npt);
    
    // Initialize RNG
    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

    Kokkos::parallel_for("initialize_cells", Npt, KOKKOS_LAMBDA(const int i) {
      xc(i, 0) = (double(i / Npthalf) + 0.5) / Npthalf;
      xc(i, 1) = (double(i % Npthalf) + 0.5) / Npthalf;
      // xc(i, 0) = 0.5;
      // xc(i, 1) = 0.5;
      xc(i, 2) = 0.5;

      dxc(i) = 0.5 / Npthalf;

      // auto generator = random_pool.get_state();
      // vc(i, 0) = normal(0.5, 0.1, generator);
      // vc(i, 1) = normal(0.5, 0.1, generator);
      // vc(i, 2) = normal(0.5, 0.1, generator);
      // random_pool.free_state(generator);
      vc(i, 0) = 0.44398917 / 2;
      vc(i, 1) = -0.65130061 / 2;
      vc(i, 2) = -0.61537073 / 2;
      sigma_vc(i) = 0.1;

      // std::cout << "vc = " << vc(i, 0) << ", " << vc(i, 1) << ", " << vc(i, 2) << std::endl;
    });

    // Create hypercube view
    Point u = {0.89572202, 0.30454387, 0.32393687};
    Point v = {-0.0235729 , -0.69502558,  0.71859847};
    Point O = {0, 0, 0};

    Config cfg;
    cfg.Npix = Npix;
    cfg.NpixVelocity = NpixVelocity;
    cfg.Npt = Npt;
    cfg.dx = 1.0;
    cfg.vmin = 0.0;
    cfg.vmax = 1.0;

    Kokkos::View<double***> cube = hypercube(xc, vc, dxc, sigma_vc, u, v, O, cfg);

    // Copy view to host
    auto cube_h0 = Kokkos::create_mirror_view(cube);
    Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace> cube_h("ypercube_C_order", cfg.Npix, cfg.Npix, cfg.NpixVelocity);
    Kokkos::deep_copy(cube_h0, cube);
    Kokkos::deep_copy(cube_h, cube_h0);

    // Write as raw binary
    std::cout << "Writing to numpy array" << std::endl;
    std::ofstream file("cube.bin", std::ios::binary);
    file.write((char*) &Npix, sizeof(int));
    file.write((char*) &NpixVelocity, sizeof(int));
    file.write((char*) &Npt, sizeof(int));

    file.write((char*) &u.x, sizeof(Point));
    file.write((char*) &v.x, sizeof(Point));

    file.write((char*) cube_h.data(), cube_h.size() * sizeof(double));

  }

  Kokkos::finalize();
}