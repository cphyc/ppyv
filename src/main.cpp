#include <math.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

typedef Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> View3D_C;
typedef Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> View2D_C;
typedef Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> View1D;
typedef py::array_t<double, py::array::c_style | py::array::forcecast> PyCArray;
typedef py::array_t<double, py::array::f_style | py::array::forcecast> PyFArray;

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

    constexpr Point() : x(0), y(0), z(0) {}
    constexpr Point(double x, double y, double z) : x(x), y(y), z(z) {}
    Point(PyCArray arr) {
      py::buffer_info buf = arr.request();
      if (buf.ndim != 1) {
        throw std::runtime_error("Input array must be 1D");
      }
      if (buf.shape[0] != 3) {
        throw std::runtime_error("Input array must have shape (3,)");
      }
      double* ptr = static_cast<double*>(buf.ptr);
      x = ptr[0];
      y = ptr[1];
      z = ptr[2];
    }

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
void direct_integrate_cube(const Point& O, const Point& Oback, const Point& u, const Point& v, const Point& w, const double weight, Kokkos::View<double***> buffer, Kokkos::View<bool**> buffer_mask) {
  const int Nx = buffer.extent(0);
  const int Ny = buffer.extent(1);

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
        Kokkos::atomic_add(&buffer(i, j, 0), (zmax - zmin) * weight);
        buffer_mask(i, j) = 1;
      } else {
        // throw std::runtime_error("Nhit must be 0 or 2");
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
  Point center = x;
  Point p000 = center - u * dx/2 - v * dx/2 - w * dx/2;
  Point p111 = center + u * dx/2 + v * dx/2 + w * dx/2;

  direct_integrate_cube(p000, p111, u * dx, v * dx, w * dx, weight, buffer, mask);
}

void hypercube(
    Kokkos::View<double**>& xc,
    Kokkos::View<double**>& vc,
    Kokkos::View<double*>& dxc,
    Kokkos::View<double*>& sigma_v,
    Kokkos::View<double*>& weight,
    const Point u,  // x-axis
    const Point v,  // y-axis
    const Point O,  // Origin
    const Config cfg,
    Kokkos::View<double***>& output
) {
  const Point w = {u.y*v.z - u.z*v.y,
                   u.z*v.x - u.x*v.z,
                   u.x*v.y - u.y*v.x};

  const double dv = cfg.vmax - cfg.vmin;
  double view_tot = 0.0;

  Kokkos::View<bool**> mask("mask", output.extent(0), output.extent(1));


  Kokkos::parallel_for("hypercube", cfg.Npt, KOKKOS_LAMBDA(const int i) {
    Point xcell = {
      (xc(i, 0) - O.x) / cfg.dx,
      (xc(i, 1) - O.y) / cfg.dx,
      (xc(i, 2) - O.z) / cfg.dx
    };
    Point vcell = {vc(i, 0), vc(i, 1), vc(i, 2)};

    cell2hypercube(xcell, dxc(i), vcell, sigma_v(i), u, v, w, weight(i), cfg.Npix, cfg.NpixVelocity, output, mask);
  });

  Kokkos::parallel_reduce("sum", cfg.Npix, KOKKOS_LAMBDA(const int i, double& view_tot) {
    for (auto j = 0; j < output.extent(1); j++) {
      view_tot += output(i, j, 0);
    }
  }, view_tot);

  std::cout << "view_tot: " << view_tot << " (" << view_tot / cfg.Npt << ")" << std::endl;
}



PyCArray compute_hypercube(
  const PyCArray xc,
  const PyCArray vc,
  const PyCArray dxc,
  const PyCArray sigma_vc,
  const PyCArray weight,
  const int Npix,
  const int NpixVelocity,
  const PyCArray u,
  const PyCArray v,
  const PyCArray O
) {
  // Check input array sizes
  py::buffer_info buf_x = xc.request(),
                  buf_v = vc.request(), 
                  buf_dx = dxc.request(),
                  buf_sigma = sigma_vc.request(),
                  buf_weight = weight.request();
  if (buf_x.ndim != 2 || buf_v.ndim != 2) {
    throw std::runtime_error("`xc` and `vc` must be 2D");
  }
  if (buf_dx.ndim != 1 || buf_sigma.ndim != 1 || buf_weight.ndim != 1) {
    throw std::runtime_error("`dxc`, `sigma_vc`, and `weight` must be 1D");
  }
  if (buf_x.shape[1] != 3 || buf_v.shape[1] != 3) {
    throw std::runtime_error("`xc` and `vc` must have shape (Npt, 3)");
  }
  if (buf_x.shape[0] != buf_v.shape[0] || buf_x.shape[0] != buf_dx.shape[0] || buf_x.shape[0] != buf_sigma.shape[0] || buf_x.shape[0] != buf_weight.shape[0]) {
    throw std::runtime_error("All input arrays must have the same length");
  }

  // Create hypercube view
  Point uu(u);
  Point vv(v);
  Point OO(O);

  Config cfg;
  cfg.Npix = Npix;
  cfg.NpixVelocity = NpixVelocity;
  cfg.Npt = xc.shape(0);
  // TODO
  cfg.dx = 1.0;
  cfg.vmin = 0.0;
  cfg.vmax = 1.0;

  std::cout << "uu: " << uu.x << " " << uu.y << " " << uu.z << std::endl;
  std::cout << "vv: " << vv.x << " " << vv.y << " " << vv.z << std::endl;

  // Output buffer
  PyCArray output = PyCArray({cfg.Npix, cfg.Npix, cfg.NpixVelocity});
  py::buffer_info buf_output = output.request();

  {
    // Create views out of numpy arrays [do not manage memory]
    double *ptr_x = static_cast<double *>(buf_x.ptr);
    double *ptr_v = static_cast<double *>(buf_v.ptr);
    double *ptr_dx = static_cast<double *>(buf_dx.ptr);
    double *ptr_sigma = static_cast<double *>(buf_sigma.ptr);
    double *ptr_w = static_cast<double *>(buf_weight.ptr);
    View2D_C xc_host(ptr_x, cfg.Npt, 3);
    View2D_C vc_host(ptr_v, cfg.Npt, 3);
    View1D dxc_host(ptr_dx, cfg.Npt);
    View1D sigma_vc_host(ptr_sigma, cfg.Npt);
    View1D weight_host(ptr_w, cfg.Npt);

    // Copy to device
    Kokkos::View<double**> xc_dev("xc", cfg.Npt, 3);
    Kokkos::View<double**> vc_dev("vc", cfg.Npt, 3);
    Kokkos::View<double*> dxc_dev("dxc", cfg.Npt);
    Kokkos::View<double*> sigma_vc_dev("sigma_v", cfg.Npt);
    Kokkos::View<double*> weight_dev("weight", cfg.Npt);

    // Copy input data to device
    {
      auto xc_tmp = Kokkos::create_mirror_view(xc_dev);
      Kokkos::deep_copy(xc_tmp, xc_host);
      Kokkos::deep_copy(xc_dev, xc_tmp);
    }
    {
      auto vc_tmp = Kokkos::create_mirror_view(vc_dev);
      Kokkos::deep_copy(vc_tmp, vc_host);
      Kokkos::deep_copy(vc_dev, vc_tmp);
    }
    Kokkos::deep_copy(dxc_dev, dxc_host);
    Kokkos::deep_copy(sigma_vc_dev, sigma_vc_host);
    Kokkos::deep_copy(weight_dev, weight_host);

    Kokkos::View<double***> cube("hypercube", cfg.Npix, cfg.Npix, cfg.NpixVelocity);
    hypercube(xc_dev, vc_dev, dxc_dev, sigma_vc_dev, weight_dev, uu, vv, OO, cfg, cube);

    // Copy output to host
    auto cube_host = Kokkos::create_mirror_view(cube);
    Kokkos::deep_copy(cube_host, cube);

    // Copy into output buffer
    double *ptr_output = static_cast<double *>(buf_output.ptr);
    View3D_C output_view(ptr_output, cfg.Npix, cfg.Npix, cfg.NpixVelocity);
    Kokkos::deep_copy(output_view, cube_host);
  }

  return output;
}

// int main(int argc, char* argv[]) {
//   Kokkos::initialize(argc, argv);
//   const int Npix = 128;
//   const int NpixVelocity = 128;
//   const int Npt = 16;
//   const int Npthalf = Npt / 2;
//   {
//     auto read_fortran_record = [](std::ifstream& file, auto* data, int N) {
//       uint32_t record_size_start, record_size_end;

//       size_t record_size = N * sizeof(decltype(data));

//       // Read record size at start
//       file.read((char*) &record_size_start, sizeof(uint32_t));
//       if (record_size_start != N * sizeof(decltype(&data))) {
//         throw std::runtime_error(
//           "Record size mismatch at start, expected "
//           + std::to_string(N * sizeof(decltype(&data)))
//           + " but got " + std::to_string(record_size_start)
//         );
//       }
//       // Read data
//       file.read((char*) data, N * sizeof(decltype(&data)));

//       // Read record size at end
//       file.read((char*) &record_size_end, sizeof(uint32_t));
//       if (record_size_end != N * sizeof(decltype(&data))) {
//         throw std::runtime_error(
//           "Record size mismatch at end, expected "
//           + std::to_string(N * sizeof(decltype(&data)))
//           + " but got " + std::to_string(record_size_end)
//         );
//       }
//     };

//     auto read_fortran = [](std::ifstream& file, auto& value) {
//       uint32_t record_size_start, record_size_end;

//       // Read record size at start
//       file.read((char*) &record_size_start, sizeof(decltype(value)));
//       if (record_size_start != sizeof(int)) {
//         throw std::runtime_error(
//           "Record size mismatch at start, expected "
//           + std::to_string(sizeof(int))
//           + " but got " + std::to_string(record_size_start)
//         );
//       }

//       // Read data
//       file.read((char*) &value, sizeof(decltype(value)));

//       // Read record size at end
//       file.read((char*) &record_size_end, sizeof(decltype(value)));
//       if (record_size_end != sizeof(int)) {
//         throw std::runtime_error(
//           "Record size mismatch at end, expected "
//           + std::to_string(sizeof(int))
//           + " but got " + std::to_string(record_size_end)
//         );
//       }
//     };

//     // Read data from file
//     std::ifstream ifile("/tmp/data.bin", std::ios::binary);
//     uint32_t Npt;
//     read_fortran(ifile, Npt);

//     std::cout << "Npt = " << Npt << std::endl;

//     // Create point view
//     Kokkos::View<double**> xc("xc", Npt, 3);
//     Kokkos::View<double**> vc("vc", Npt, 3);
//     Kokkos::View<double*> dxc("dxc", Npt);
//     Kokkos::View<double*> sigma_vc("sigma_v", Npt);
//     Kokkos::View<double*> weight("weight", Npt);

//     {
//       auto xc_host = Kokkos::create_mirror_view(xc);
//       auto vc_host = Kokkos::create_mirror_view(vc);
//       auto dxc_host = Kokkos::create_mirror_view(dxc);
//       auto sigma_vc_host = Kokkos::create_mirror_view(sigma_vc);

//       Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> xc_host_F = Kokkos::create_mirror_view(xc_host);
//       Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vc_host_F = Kokkos::create_mirror_view(vc_host);

//       // Read positions [note, we use Fortran order]
//       read_fortran_record(ifile, Kokkos::subview(xc_host_F, Kokkos::ALL, 0).data(), Npt);
//       read_fortran_record(ifile, Kokkos::subview(xc_host_F, Kokkos::ALL, 1).data(), Npt);
//       read_fortran_record(ifile, Kokkos::subview(xc_host_F, Kokkos::ALL, 2).data(), Npt);
//       read_fortran_record(ifile, dxc_host.data(), Npt);
//       read_fortran_record(ifile, Kokkos::subview(vc_host_F, Kokkos::ALL, 0).data(), Npt);
//       read_fortran_record(ifile, Kokkos::subview(vc_host_F, Kokkos::ALL, 1).data(), Npt);
//       read_fortran_record(ifile, Kokkos::subview(vc_host_F, Kokkos::ALL, 2).data(), Npt);
//       read_fortran_record(ifile, sigma_vc_host.data(), Npt);

//       // Close file
//       ifile.close();

//       // Copy from Fortran to C order
//       Kokkos::deep_copy(xc_host, xc_host_F);
//       Kokkos::deep_copy(vc_host, vc_host_F);

//       // Copy to device
//       Kokkos::deep_copy(xc, xc_host);
//       Kokkos::deep_copy(vc, vc_host);
//       Kokkos::deep_copy(dxc, dxc_host);
//       Kokkos::deep_copy(sigma_vc, sigma_vc_host);
//       std::cout << "xp = " << xc_host_F(1000, 0) << ", " << xc_host_F(1000, 1) << ", " << xc_host_F(1000, 2) << std::endl;
//     }

//     std::cout << "HERE!" << std::endl;


//     // // Initialize RNG
//     // Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

//     // Kokkos::parallel_for("initialize_cells", Npt, KOKKOS_LAMBDA(const int i) {
//     //   xc(i, 0) = (double(i / Npthalf) + 0.5) / Npthalf;
//     //   xc(i, 1) = (double(i % Npthalf) + 0.5) / Npthalf;
//     //   // xc(i, 0) = 0.5;
//     //   // xc(i, 1) = 0.5;
//     //   xc(i, 2) = 0.5;

//     //   dxc(i) = 0.5 / Npthalf;

//     //   // auto generator = random_pool.get_state();
//     //   // vc(i, 0) = normal(0.5, 0.1, generator);
//     //   // vc(i, 1) = normal(0.5, 0.1, generator);
//     //   // vc(i, 2) = normal(0.5, 0.1, generator);
//     //   // random_pool.free_state(generator);
//     //   vc(i, 0) = 0.44398917 / 2;
//     //   vc(i, 1) = -0.65130061 / 2;
//     //   vc(i, 2) = -0.61537073 / 2;
//     //   sigma_vc(i) = 0.1;

//     //   // std::cout << "vc = " << vc(i, 0) << ", " << vc(i, 1) << ", " << vc(i, 2) << std::endl;
//     // });

//     // Create hypercube view
//     Point u = {0.89572202, 0.30454387, 0.32393687};
//     Point v = {-0.0235729 , -0.69502558,  0.71859847};
//     Point O = {0, 0, 0};

//     Config cfg;
//     cfg.Npix = Npix;
//     cfg.NpixVelocity = NpixVelocity;
//     cfg.Npt = Npt;
//     cfg.dx = 1.0;
//     cfg.vmin = 0.0;
//     cfg.vmax = 1.0;

//     Kokkos::View<double***> cube("hypercube", cfg.Npix, cfg.Npix, cfg.NpixVelocity);
//     hypercube(xc, vc, dxc, sigma_vc, weight, u, v, O, cfg, cube);

//     // Copy view to host
//     auto cube_h0 = Kokkos::create_mirror_view(cube);
//     Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace> cube_h = cube_h0;
//     Kokkos::deep_copy(cube_h0, cube);
//     Kokkos::deep_copy(cube_h, cube_h0);

//     // Write as raw binary
//     std::cout << "Writing to numpy array" << std::endl;
//     std::ofstream file("cube.bin", std::ios::binary);
//     file.write((char*) &Npix, sizeof(int));
//     file.write((char*) &NpixVelocity, sizeof(int));
//     file.write((char*) &Npt, sizeof(int));

//     file.write((char*) &u.x, sizeof(Point));
//     file.write((char*) &v.x, sizeof(Point));

//     file.write((char*) cube_h.data(), cube_h.size() * sizeof(double));

//   }
// }

void initialize(){
  Kokkos::initialize();
}

void finalize(){
  Kokkos::finalize();
}

PYBIND11_MODULE(ppv, m) {
  m.doc() = "PPV module";

  m.def("compute_hypercube", &compute_hypercube, "Compute hypercube");
  m.def("initialize", &initialize, "Initialize Kokkos");
  m.def("finalize", &finalize, "Finalize Kokkos");
}