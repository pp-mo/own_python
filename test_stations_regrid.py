'''
Created on 5 Apr 2015

@author: ppeglar
'''
import numpy as np
import numpy.random as random

import iris
import iris.analysis
import iris.analysis.trajectory
import iris.cube
import iris.coords
import iris.coord_systems
import iris.fileformats.pp

from pp_utils import TimedBlock

cs_pc = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)

def dummy_field(shape=(1400, 1000)):
    ny, nx = shape
    data = np.arange(nx * ny).reshape(shape)
    surface_press_cube = iris._cube.Cube(data, long_name='test_values')
    lons = np.linspace(-180.0, 180.0, nx, endpoint=False)
    lats = np.linspace(-90.0, 90.0, ny, endpoint=True)
    co_x = iris.coords.DimCoord(lons, standard_name='longitude',
                                units='degrees', coord_system=cs_pc)
    co_y = iris.coords.DimCoord(lats, standard_name='latitude',
                                units='degrees', coord_system=cs_pc)
    surface_press_cube.add_dim_coord(co_y, 0)
    surface_press_cube.add_dim_coord(co_x, 1)
    return surface_press_cube


N_STATIONS = 743
# N_STATIONS = 73
# N_STATIONS = 5

STATIONS_SEED = 743


def station_lats_and_lons(n_stations=N_STATIONS):
    random.seed(STATIONS_SEED)
    lats = random.uniform(-90.0, 90.0, N_STATIONS)
    lons = random.uniform(-180.0, 180.0, N_STATIONS)
    return lats, lons


def show_result(computed_results, timedblock):
    print 'computed_results = ', computed_results[:4], ' ...'
    print 'time = ', timedblock.seconds()


def run_main():
    surface_press_cube = dummy_field()
    st_lats, st_lons = station_lats_and_lons()

    scheme = iris.analysis.Linear()

    print 'Testing with {} stations'.format(N_STATIONS)

    print
    print 'Method #1 : regrid onto 1x1 _cube at each station.'
    with TimedBlock() as tb_1:
        target_grid = dummy_field((1, 1))
        x_coord = target_grid.coord('longitude')
        y_coord = target_grid.coord('latitude')
        results_1 = np.empty(N_STATIONS)
        for i_station, (lat, lon) in enumerate(zip(st_lats, st_lons)):
            x_coord.points = [lon]
            y_coord.points = [lat]
            regridded_cube = surface_press_cube.regrid(target_grid, scheme)
            results_1[i_station] = regridded_cube.data[0]
    show_result(results_1, tb_1)

    print
    print 'Method #2 : trajectory regridding.'
    with TimedBlock() as tb_2:
        sample_points = [('longitude', st_lons), ('latitude', st_lats)]
        regridded_cube = iris.analysis.trajectory.interpolate(
            surface_press_cube, sample_points)
        results_2 = regridded_cube.data
    show_result(results_2, tb_2)
    assert np.allclose(results_2, results_1)

    print
    print 'Method #3 : squared XY regridding.'
    with TimedBlock() as tb_3:
        sorted_lons = np.sort(st_lons)
        lon_inds = np.argsort(st_lons)
        sorted_lats = np.sort(st_lats)
        lat_inds = np.argsort(st_lats)
        rev_lon_inds = np.empty(N_STATIONS, dtype=int)
        rev_lon_inds[lon_inds] = np.arange(N_STATIONS)
        rev_lat_inds = np.empty(N_STATIONS, dtype=int)
        rev_lat_inds[lat_inds] = np.arange(N_STATIONS)
        target_grid = dummy_field((N_STATIONS, N_STATIONS))
        target_grid.coord('longitude').points = sorted_lons
        target_grid.coord('latitude').points = sorted_lats
        regridded_cube = surface_press_cube.regrid(target_grid, scheme)
        results_3 = regridded_cube.data[rev_lat_inds, rev_lon_inds]
    show_result(results_3, tb_3)
    assert np.allclose(results_3, results_1)


if __name__ == "__main__":
    run_main()
