import pytest
from ppyv.ppyv import compute_hypercube, initialize, finalize
import numpy as np


def setup_module():
    initialize()


def teardown_module():
    finalize()


def test_conservation_big_cube():
    x = np.array([[0.5, 0.5, 0.5]])
    # Just to make sure we're not *exactly* on the grid
    v = np.array([[0.5, 0.5, 0.5]]) + 1 / 128
    dx = np.array([0.6789])
    dv = np.array([1e-10])

    theta = 0.123
    uu = np.array([np.cos(theta), np.sin(theta), 0])
    vv = np.array([-np.sin(theta), np.cos(theta), 0])

    cube = compute_hypercube(
        pos=x,
        vel=v,
        dx=dx,
        sigma_v=dv,
        weight=np.array([1]),
        Npix=64,
        Npix_velocity=64,
        u=uu,
        v=vv,
        O=np.array([0.5, 0.5, 0.5], dtype="d"),
        width=1,
        vmin=0,
        vmax=1,
    )

    # Check that collapsing along velocity gives back dx
    np.testing.assert_allclose(cube.sum(axis=-1).max(), dx)
    assert cube.min() == 0

    # Check that we are full of zeroes in v direction, except for
    # the halfpoint
    np.testing.assert_allclose(cube[..., :32], 0)
    np.testing.assert_allclose(cube[..., 33:], 0)


def test_conservation_one_pixel():
    x = np.array([[0.5] * 3])
    # Just to make sure we're not *exactly* on the grid
    v = np.array([[0.5, 0.5, 0.5]]) + 1 / 128
    dx = np.array([1 / 64])
    x += dx / 2
    dv = np.array([1e-10])

    theta = 0.123
    uu = np.array([np.cos(theta), np.sin(theta), 0])
    vv = np.array([-np.sin(theta), np.cos(theta), 0])

    cube = compute_hypercube(
        pos=x,
        vel=v,
        dx=dx,
        sigma_v=dv,
        weight=np.array([1]),
        Npix=64,
        Npix_velocity=64,
        u=uu,
        v=vv,
        O=np.array([0.5, 0.5, 0.5], dtype="d"),
        width=1,
        vmin=0,
        vmax=1,
    )

    # Check that collapsing along velocity gives back dx
    np.testing.assert_allclose(cube.sum(axis=-1).max(), dx)
    assert cube.min() == 0

    # Check that we are full of zeroes in v direction, except for
    # the halfpoint
    np.testing.assert_allclose(cube[..., :32], 0)
    np.testing.assert_allclose(cube[..., 33:], 0)


def test_conservation_subpixel():
    x = np.array([[0.5] * 3])
    # Just to make sure we're not *exactly* on the grid
    v = np.array([[0.5, 0.5, 0.5]]) + 1 / 128
    dx = np.array([1 / 64])
    x += dx / 2
    dx[:] = 1 / 128
    dv = np.array([1e-10])

    theta = 0.123
    uu = np.array([np.cos(theta), np.sin(theta), 0])
    vv = np.array([-np.sin(theta), np.cos(theta), 0])

    cube = compute_hypercube(
        pos=x,
        vel=v,
        dx=dx,
        sigma_v=dv,
        weight=np.array([1]),
        Npix=64,
        Npix_velocity=64,
        u=uu,
        v=vv,
        O=np.array([0.5, 0.5, 0.5], dtype="d"),
        width=1,
        vmin=0,
        vmax=1,
    )

    # Check that collapsing along velocity gives back dx
    np.testing.assert_allclose(cube.sum(axis=-1).max(), dx / 4)
    assert cube.min() == 0

    # Check that we are full of zeroes in v direction, except for
    # the halfpoint
    np.testing.assert_allclose(cube[..., :32], 0)
    np.testing.assert_allclose(cube[..., 33:], 0)


def test_xyz_permutations():
    x = np.array([[0.5, 0.5, 0.5]])
    # Just to make sure we're not *exactly* on the grid
    v = np.array([[0.5, 0.5, 0.5]]) + 1 / 128
    dx = np.array([0.6789])
    dv = np.array([1e-10])

    cubes = []
    for i in range(3):
        uu = np.array([0, 0, 0])
        uu[i] = 1
        vv = np.array([0, 0, 0])
        vv[(i + 1) % 3] = 1

        cube = compute_hypercube(
            pos=x,
            vel=v,
            dx=dx,
            sigma_v=dv,
            weight=np.array([1]),
            Npix=64,
            Npix_velocity=64,
            u=uu,
            v=vv,
            O=np.array([0.5, 0.5, 0.5], dtype="d"),
            width=1,
            vmin=0,
            vmax=1,
        )
        cubes.append(cube)

    np.testing.assert_allclose(cubes[0], cubes[1])
    np.testing.assert_allclose(cubes[1], cubes[2])


def random_directions_scales():
    np.random.seed(18021992)
    random_uv = np.zeros((100, 2, 3))
    random_scale = np.random.uniform(1e-5, 1 / np.sqrt(3), size=len(random_uv))
    for i in range(len(random_uv)):
        # Draw normals from a sphere
        uu, vv = np.random.normal(size=(2, 3))
        uu /= np.linalg.norm(uu)
        vv = vv - np.dot(uu, vv) * uu
        vv /= np.linalg.norm(vv)
        random_uv[i] = uu, vv

    return zip(random_scale, random_uv[:, 0, :], random_uv[:, 1, :])


# Parametrized test
@pytest.mark.parametrize("scale, uu, vv", random_directions_scales())
def test_random_permutations(scale, uu, vv):
    x = np.array([[0.5, 0.5, 0.5]])
    # Just to make sure we're not *exactly* on the grid
    v = np.array([[0.5] * 3]) + 1 / 128
    dv = np.array([1e-10])
    Npix = 64

    dx = np.array([scale])
    cube = compute_hypercube(
        pos=x,
        vel=v,
        dx=dx,
        sigma_v=dv,
        weight=np.array([1]) / Npix / Npix / dx**3,
        Npix=Npix,
        Npix_velocity=32,
        u=uu,
        v=vv,
        O=np.array([0.5, 0.5, 0.5], dtype="d"),
        width=1,
        vmin=0,
        vmax=1,
    )

    print(uu, vv, scale)
    np.testing.assert_allclose(cube.sum(), 1, atol=5e-2)
