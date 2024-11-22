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
    x = np.array([[0.5]*3])
    # Just to make sure we're not *exactly* on the grid
    v = np.array([[0.5, 0.5, 0.5]]) + 1 / 128
    dx = np.array([1/64])
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
    x = np.array([[0.5]*3])
    # Just to make sure we're not *exactly* on the grid
    v = np.array([[0.5, 0.5, 0.5]]) + 1 / 128
    dx = np.array([1/64])
    x += dx / 2
    dx[:] = 1/128
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
    np.testing.assert_allclose(cube.sum(axis=-1).max(), dx/4)
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
        vv[(i+1)%3] = 1

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
