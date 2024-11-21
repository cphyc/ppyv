# PPyV

The PPyV package allows to project 3D AMR datasets onto a xyv cube.
The package is written in C++ and uses:
- the [Kokkos](https://github.com/kokkos/kokkos) library for parallelization on GPUs and CPUs,
- the [pybind11](https://github.com/pybind/pybind11) library to create a Python interface.

## Installation

### Requirements

The PPyV package requires:
- a C++17 compiler (GCC, clang, ...),
- cmake,
- a compiler for the GPU if compiling for that target.

## Installation

```bash
# Clone the repository
git clone https://github.com/cphyc/ppyv.git --recurse-submodules

# Create a build directory
mkdir build

# Go to the build directory
cd build

# Configure the project in serial mode
cmake -DCMAKE_BUILD_TYPE=Release ..
# or parallel mode
cmake -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON ..
# or with CUDA support
cmake -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON ..
# or with HIP support
cmake -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_HIP=ON ..

# Build the project
make

# Test import
python -c 'import ppyv'
```

## Note on compiling with CUDA

Refer to https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#host-compiler-support-policy to check the compatibility of your compiler with CUDA. For example, CUDA 12.6 supports only GCC 6.x to 13.2 or Clang 7.x to 18.0.

# Usage

```python
from matplotlib import pyplot as plt
import numpy as np

import yt
import ppyv


# Create velocity dispersion field
@yt.derived_field(
    name=("gas", "velocity_dispersion"),
    units="km/s",
    sampling_type="cell",
    validators=[yt.ValidateSpatial(ghost_zones=1)],
)
def velocity_dispersion(field, data):
    m = (data["cell_mass"].to("Msun")).d
    px = (m * data["velocity_x"].to("km/s")).d
    py = (m * data["velocity_y"].to("km/s")).d
    pz = (m * data["velocity_z"].to("km/s")).d

    output = np.zeros_like(m)
    for i in range(1, output.shape[0] - 1):
        for j in range(1, output.shape[1] - 1):
            for k in range(1, output.shape[2] - 1):
                # Use 27 neighbours to estimate the velocity dispersion
                sl = slice(i - 1, i + 2), slice(j - 1, j + 2), slice(k - 1, k + 2)
                mtot = m[sl].sum(axis=(0, 1, 2))
                varx = np.var(px[sl], axis=(0, 1, 2))
                vary = np.var(py[sl], axis=(0, 1, 2))
                varz = np.var(pz[sl], axis=(0, 1, 2))
                output[i, j, k] = np.sqrt(varx + vary + varz) / mtot

    return data.apply_units(output, "km/s")


Npix = 256
Npix_velocity = 256

# Load example dataset
ds = yt.load_sample("output_00080")

# Find the densest cell
ad = ds.all_data()
center = ds.arr(ad.argmax("density"))

# Create a sphere around it
r = ds.quan(400, "kpc")
sp = ds.sphere(center, r)
sp["velocity_dispersion"]
spp = ds.sphere(center, (5, "kpc"))

bulk_velocity = spp.quantities.bulk_velocity(use_gas=True, use_particles=False)
normal = sp.quantities.angular_momentum_vector(use_gas=True, use_particles=False)
normal = (normal / np.linalg.norm(normal)).d

# Extract data
xc = (np.stack([(sp[k]).to("kpc") for i, k in enumerate("xyz")], axis=1)).d
vc = np.stack([(sp[f"velocity_{k}"] - bulk_velocity[i]).to("km/s") for i, k in enumerate("xyz")], axis=1).d
dx = sp["dx"].to("kpc").d
sigma_v = sp["velocity_dispersion"].to("km/s").d
rho = sp["density"].to("mp/cm**3").d

# Parameters for the plot
vmin = -50
vmax =50

## Compute normal vectors

w = normal
u = np.cross(w, [1, 0, 0])
u /= np.linalg.norm(u)
v = np.cross(w, u)
v /= np.linalg.norm(v)
O = sp.center.to("kpc").d

print("================")
print(u, v, w)
print("================")

cube = ppyv.compute_hypercube(
    pos=xc,
    vel=vc,
    dx=dx,
    sigma_v=sigma_v,
    weight=rho,
    Npix=Npix,
    Npix_velocity=Npix_velocity,
    u=u,
    v=v,
    O=O,
    width=2 * r.d,
    vmin=vmin,
    vmax=vmax,
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

norm = plt.matplotlib.colors.LogNorm()
axes[0].imshow(
    cube.sum(axis=2).T,
    origin="lower",
    norm=norm,
    extent=(-r.d, r.d, -r.d, r.d),
    cmap="plasma",
)
axes[0].set(
    xlabel=f"x [{r.units}]",
    ylabel=f"y [{r.units}]",
)
sl = slice(Npix//4, 3*Npix//4)
axes[1].imshow(
    cube[:, sl, :].sum(axis=1).T,
    origin="lower",
    extent=(-r.d, r.d, vmin, vmax),
    norm=norm,
    cmap="plasma",
)
axes[1].set_aspect("auto")
axes[1].set(
    xlabel=f"x [{r.units}]",
    ylabel="v [km/s]",
)
axes[2].imshow(
    cube[sl, :, :].sum(axis=0).T,
    origin="lower",
    extent=(-r.d, r.d, vmin, vmax),
    norm=norm,
    cmap="plasma",
)
axes[2].set_aspect("auto")
axes[2].set(
    xlabel=f"y [{r.units}]",
    ylabel="v [km/s]",
)
# Create axis for colorbar
plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap="plasma"),
    ax=axes,
    orientation="vertical",
    fraction=0.05,
    aspect=20,
)
plt.show()
```
