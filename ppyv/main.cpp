#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

typedef Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
    View3D_C;
typedef Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
    View2D_C;
typedef Kokkos::View<double *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
    View1D;
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

bool _finalize_kokkos() {
  if (!Kokkos::is_initialized())
    return false;
  Kokkos::Tools::Experimental::set_deallocate_data_callback(nullptr);
  py::module gc = py::module::import("gc");
  gc.attr("collect")();
  Kokkos::finalize();
  return true;
}

void _finalize_kokkos_void() { _finalize_kokkos(); }

// Initialize kokkos
bool _initialize_kokkos() {
  if (Kokkos::is_initialized())
    return false;

  // python system module
  py::module sys = py::module::import("sys");
  // get the arguments for python system module
  py::object args = sys.attr("argv");
  auto argv = args.cast<py::list>();
  int _argc = argv.size();
  char **_argv = new char *[argv.size()];
  for (int i = 0; i < _argc; ++i) {
    auto _args = argv[i].cast<std::string>();
    if (_args == "--") {
      for (int j = i; j < _argc; ++j)
        _argv[i] = nullptr;
      _argc = i;
      break;
    }
    _argv[i] = strdup(_args.c_str());
  }
  Kokkos::initialize(_argc, _argv);
  for (int i = 0; i < _argc; ++i)
    free(_argv[i]);
  delete[] _argv;
  return true;
};

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
    double *ptr = static_cast<double *>(buf.ptr);
    x = ptr[0];
    y = ptr[1];
    z = ptr[2];
  }

  KOKKOS_INLINE_FUNCTION
  constexpr Point operator+(const Point &p) const {
    return {x + p.x, y + p.y, z + p.z};
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION constexpr Point operator+(const T v) const {
    return {x + v, y + v, z + v};
  }

  KOKKOS_INLINE_FUNCTION
  constexpr Point operator-(const Point &p) const {
    return {x - p.x, y - p.y, z - p.z};
  }

  KOKKOS_INLINE_FUNCTION
  constexpr Point operator*(const double &f) const {
    return {x * f, y * f, z * f};
  }

  KOKKOS_INLINE_FUNCTION
  constexpr Point operator/(const double &f) const { return *this * (1 / f); }
};

KOKKOS_INLINE_FUNCTION
constexpr double dot2d(const Point &a, const Point &b) {
  return a.x * b.x + a.y * b.y;
};

KOKKOS_INLINE_FUNCTION
constexpr double det2d(const Point &a, const Point &b) {
  return a.x * b.y - a.y * b.x;
}

KOKKOS_INLINE_FUNCTION
constexpr double dot3d(const Point &a, const Point &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
};

KOKKOS_INLINE_FUNCTION
bool check_in_parallelogram(const Point &PA, const Point &PQ, const Point &PR,
                            const int signPQ, const int signPR, double out[2]) {
  double det_PQR = det2d(PQ, PR);

  if (det_PQR == 0) {
    out[0] = 0;
    out[1] = 0;
    return false;
  } else {
    out[0] = -det2d(PA, PQ) / det_PQR;
    out[1] = det2d(PA, PR) / det_PQR;

    if ((0 <= signPQ * out[0]) & (1 >= signPQ * out[0]) &
        (0 <= signPR * out[1]) & (1 >= signPR * out[1])) {
      return true;
    } else {
      out[0] = -1;
      out[1] = -1;
      return false;
    }
  }
}

KOKKOS_INLINE_FUNCTION
void kernel_gaussian(const int i, const int j, const int N, const double mean,
                     const double std, const double w,
                     Kokkos::View<double ***> &out) {
  const double mm = mean * N, ss = std * N;
  const double one_over_sqrt2std = 1 / (sqrt(2) * std * N);
  int kmin = std::max(int(floor(mm - 4 * ss)), 0),
      kmax = std::min(int(ceil(mm + 4 * ss)), N);

  if (kmin + 1 >= kmax) {
    Kokkos::atomic_add(&out(i, j, kmin), w);
    return;
  }

  for (auto k = kmin; k < kmax; ++k) {
    Kokkos::atomic_add(&out(i, j, k),
                       w / 2 *
                           (std::erf((k + 1 - mm) * one_over_sqrt2std) -
                            std::erf((k - mm) * one_over_sqrt2std)));
  }
}

KOKKOS_INLINE_FUNCTION
void direct_integrate_cube(const Point &O, const Point &Oback, const Point &u,
                           const Point &v, const Point &w, const double weight,
                           const double vel, const double sigma_v,
                           const int NpixVelocity, const double dx,
                           Kokkos::View<double ***> buffer) {
  const int Nx = buffer.extent(0);
  const int Ny = buffer.extent(1);

  Point X({0, 0, 0});
  Point OfrontA({0, 0, 0}), ObackA({0, 0, 0});

  const double inv_dx = 1. / Nx;
  const double inv_dy = 1. / Ny;
  double nm[2] = {0, 0};

  auto minMax = [](const double O, const double Oback, const double u,
                   const double v, const double w, const int N) {
    const auto [xmin, xmax] = std::minmax(
        {O, O + u, O + v, O + w, Oback, Oback - u, Oback - v, Oback - w});
    return std::make_pair(int(floor((xmin > 0 ? xmin : 0) * N)),
                          int(ceil((xmax < 1 ? xmax : 1) * N)));
  };

  auto [imin, imax] = minMax(O.x, Oback.x, u.x, v.x, w.x, Nx);
  auto [jmin, jmax] = minMax(O.y, Oback.y, u.y, v.y, w.y, Ny);

  // Special case: cell spans a single pixel
  // Note: if we *exactly* hit a pixel edge, imin == imax
  if (((0 <= imin) && (imin + 1 >= imax) && (imax < Nx)) &&
      ((0 <= jmin) && (jmin + 1 >= jmax) && (jmax < Ny))) {
    kernel_gaussian(imin, jmin, NpixVelocity, vel, sigma_v,
                    weight * dx * dx * dx * Nx * Ny, buffer);
    // buffer(imin, jmin, 0) += weight;
    return;
  }

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
        kernel_gaussian(i, j, NpixVelocity, vel, sigma_v,
                        (zmax - zmin) * weight, buffer);
      } else {
        // throw std::runtime_error("Nhit must be 0 or 2");
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
double cross(const Point &a, const Point &b, const Point &c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
};

// Check if point (px, py) is inside the quadrangle defined by (p0, p1, p2, p3)
// using the cross product method.
KOKKOS_INLINE_FUNCTION
double locateOnQuad(const Point &p, const Point &u, const Point &v,
                    const double dx) {
  // Calculate position of point p in (Ou)x(Ov) plane
  // We are in the face if tx and ty are in [-0.5, 0.5]
  const double tx = dot2d(p, u) / dx;
  const double ty = dot2d(p, v) / dx;

  // If all cross products have the same sign, the point is inside
  if (tx <= 0)
    return 0;
  else if (tx >= 1)
    return 0;
  else if (ty <= 0)
    return 0;
  else if (ty >= 1)
    return 0;
  else
    return tx * u.z * dx + ty * v.z * dx;
}

// KOKKOS_INLINE_FUNCTION
// void cell2hypercube(const Point &x, const double dx, const double &vel_los,
//                     const double sigma_v, const Point &u, const Point &v,
//                     const Point &w, const double weight, const int Npix,
//                     const int NpixVelocity,
//                     // Kokkos::View<double ***> buffer,
//                     Kokkos::Experimental::ScatterView<double ***> buffer,
//                     Kokkos::View<bool **> mask) {

// }

void hypercube(Kokkos::View<double **> &xc, Kokkos::View<double **> &vc,
               Kokkos::View<double *> &dxc, Kokkos::View<double *> &sigma_v,
               Kokkos::View<double *> &weight,
               const Point u, // x-axis
               const Point v, // y-axis
               const Point O, // Origin
               const Config cfg, Kokkos::View<double ***> &output) {
  const Point w = {u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z,
                   u.x * v.y - u.y * v.x};

  const double dv = cfg.vmax - cfg.vmin;

  // Axes of AABB in rotated frame
  Point uu = {u.x, v.x, w.x};
  Point vv = {u.y, v.y, w.y};
  Point ww = {u.z, v.z, w.z};

  Kokkos::parallel_for(
      "hypercube", cfg.Npt, KOKKOS_LAMBDA(const int i) {
        // Compute position relative to center
        Point xcell = {(xc(i, 0) - O.x) / cfg.dx, (xc(i, 1) - O.y) / cfg.dx,
                       (xc(i, 2) - O.z) / cfg.dx};
        // Rotate and shift to [0, 1]
        xcell = {dot3d(xcell, u) + 0.5, dot3d(xcell, v) + 0.5,
                 dot3d(xcell, w) + 0.5};

        Point vcell = {vc(i, 0), vc(i, 1), vc(i, 2)};
        const double vlos = dot3d(vcell, w);

        const double ddx = dxc(i) / cfg.dx;
        Point p000 = xcell - uu * ddx / 2 - vv * ddx / 2 - ww * ddx / 2;
        Point p111 = xcell + uu * ddx / 2 + vv * ddx / 2 + ww * ddx / 2;

        direct_integrate_cube(p000, p111, uu * ddx, vv * ddx, ww * ddx,
                              weight(i), (vlos - cfg.vmin) / dv,
                              sigma_v(i) / dv, cfg.NpixVelocity, ddx, output);
      });
}

PyCArray compute_hypercube(const PyCArray xc, const PyCArray vc,
                           const PyCArray dxc, const PyCArray sigma_vc,
                           const PyCArray weight, const int Npix,
                           const int NpixVelocity, const PyCArray u,
                           const PyCArray v, const PyCArray O, const double dx,
                           const double vmin, const double vmax) {

  // Initialize Kokkos
  _initialize_kokkos();

  // Check input array sizes
  py::buffer_info buf_x = xc.request(), buf_v = vc.request(),
                  buf_dx = dxc.request(), buf_sigma = sigma_vc.request(),
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
  if (buf_x.shape[0] != buf_v.shape[0] || buf_x.shape[0] != buf_dx.shape[0] ||
      buf_x.shape[0] != buf_sigma.shape[0] ||
      buf_x.shape[0] != buf_weight.shape[0]) {
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

  cfg.dx = dx;
  cfg.vmin = vmin;
  cfg.vmax = vmax;

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
    Kokkos::View<double **> xc_dev("xc", cfg.Npt, 3);
    Kokkos::View<double **> vc_dev("vc", cfg.Npt, 3);
    Kokkos::View<double *> dxc_dev("dxc", cfg.Npt);
    Kokkos::View<double *> sigma_vc_dev("sigma_v", cfg.Npt);
    Kokkos::View<double *> weight_dev("weight", cfg.Npt);

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

    Kokkos::View<double ***> cube("hypercube", cfg.Npix, cfg.Npix,
                                  cfg.NpixVelocity);
    hypercube(xc_dev, vc_dev, dxc_dev, sigma_vc_dev, weight_dev, uu, vv, OO,
              cfg, cube);

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

PYBIND11_MODULE(ppyv, m) {
  m.doc() = "PPyV module";

  // Finalize kokkos
  auto _finalize = []() {
    if (!Kokkos::is_initialized())
      return false;
    Kokkos::Tools::Experimental::set_deallocate_data_callback(nullptr);
    py::module gc = py::module::import("gc");
    gc.attr("collect")();
    Kokkos::finalize();
    return true;
  };

  std::atexit(_finalize_kokkos_void);

  m.def("compute_hypercube", &compute_hypercube, "Compute hypercube", "pos"_a,
        "vel"_a, "dx"_a, "sigma_v"_a, "weight"_a, "Npix"_a, "Npix_velocity"_a,
        "u"_a, "v"_a, "O"_a, "width"_a, "vmin"_a, "vmax"_a);

  m.def("initialize", _initialize_kokkos, "Initialize Kokkos");
  m.def("finalize", _finalize_kokkos, "Finalize Kokkos");
}
