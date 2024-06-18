from dataclasses import dataclass
from datetime import datetime, timedelta
import netCDF4
import numpy as np
from numbers import Real
from os import PathLike
from pathlib import Path
from typing import Union
import xarray as xr

# bit.sea imports
from basins import Basin


class AllPointsCropped(ValueError):
    pass


def from_datetime_to_datetime64(dt):
    return np.datetime64(dt.strftime('%Y-%m-%dT%H:%M:%S'))


@dataclass
class TimeSeries:
    dataset: str
    variable: str
    basin: Basin
    data: xr.DataArray

    @property
    def x(self) -> np.ndarray:
        return self.data.time.to_numpy()

    @property
    def y(self) -> np.ndarray:
        return self.data.to_numpy()

    def crop_time(self, start_time=None, end_time=None):
        old_time_steps = self.x
        start_index = 0

        if isinstance(start_time, datetime):
            start_time = from_datetime_to_datetime64(start_time)
        if isinstance(end_time, datetime):
            end_time = from_datetime_to_datetime64(end_time)

        if start_time is not None:
            while start_index < len(old_time_steps) and \
                    old_time_steps[start_index] < start_time:
                start_index += 1
            if old_time_steps[start_index] < start_time:
                raise AllPointsCropped(
                    'No points survived after the cropping: they are all '
                    'before {}'.format(start_time)
                )

        end_index = len(old_time_steps)
        if end_time is not None:
            while end_index > 1 and old_time_steps[end_index - 1] > end_time:
                end_index -= 1
            if old_time_steps[end_index - 1] > end_time:
                raise AllPointsCropped(
                    'No points survived after the cropping: they are all '
                    'after {}'.format(end_time)
                )

        new_data = self.data[start_index:end_index]

        return TimeSeries(
            dataset=self.dataset,
            variable=self.variable,
            basin=self.basin,
            data=new_data
        )

    def moving_average(self, window_size: int | np.timedelta64 | timedelta):
        if isinstance(window_size, timedelta):
            window_size = np.timedelta64(int(window_size.total_seconds()), 's')
        if isinstance(window_size, np.timedelta64):
            time_steps = self.x
            time_index = 0
            for time_index, current_time in enumerate(time_steps):
                if current_time - time_steps[0] >= window_size:
                    break
            if time_index == 0:
                raise ValueError(
                    'window_size is too big or not time steps inside the '
                    'current object'
                )
            window_size = time_index

        if window_size <= 0:
            raise ValueError(
                'If time_index is a number, it must be a positive integer'
            )

        # How many points do we lose because of the average (the ones on the
        # boundary of the time series)
        lost_points = window_size - 1

        original_times = self.x

        if window_size == 1:
            new_time_steps = original_times
        elif window_size % 2 == 1:
            lost_per_side = lost_points // 2
            new_time_steps = original_times[lost_per_side: - lost_per_side]
        else:
            lost_per_side = (lost_points - 1) // 2
            new_time_steps = []
            averaged_times = original_times[lost_per_side:- lost_per_side]
            for i, t1, in enumerate(averaged_times[:-1]):
                t2 = averaged_times[i + 1]
                delta_time = t2 - t1
                new_time_steps.append(t1 + delta_time // 2)
            new_time_steps = np.array(new_time_steps)
            if window_size != 2:
                assert len(new_time_steps) == len(original_times) - lost_points
            else:
                assert len(new_time_steps) == len(original_times) - 1

        original_data = self.y

        v1 = np.cumsum(original_data, axis=0)[window_size - 1:]

        v2 = np.zeros_like(v1)
        v2[1:] = np.cumsum(original_data, axis=0)[: -window_size]

        moving_average = (v1 - v2) / window_size

        new_data = xr.DataArray(
            moving_average,
            dims=('time',),
            coords={'time': new_time_steps, 'depth': self.data.depth}
        )

        return TimeSeries(
            dataset=self.dataset,
            variable=self.variable,
            basin=self.basin,
            data=new_data
        )


class TimeSeriesDataSet:
    def __init__(self, name: str, data_path: Union[PathLike, str]):
        self.name: str = name
        self._path: Path = Path(data_path)

        variables = set()
        for f_name in self._path.glob('*.nc'):
            variables.add(f_name.stem)

        self._variables: frozenset[str] = frozenset(variables)

        basin_uuids = set()
        costals = set()
        for variable in self._variables:
            file_name = variable + '.nc'
            with netCDF4.Dataset(self._path / file_name, 'r') as f:
                f_basin_uuids = [t.strip() for t in f.sub___list.split(',')]
                f_costals = [t.strip() for t in f.coast_list.split(',')]
                basin_uuids.update(f_basin_uuids)
                costals.update(f_costals)

        basins = []
        for basin_uuid in basin_uuids:
            basins.append(Basin.load_from_uuid(basin_uuid))

        self._basins: tuple[Basin] = tuple(basins)
        self._costals: tuple[str] = tuple(costals)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def variable_names(self) -> frozenset[str]:
        return self._variables

    def get_time_series(self, variable_name: str, basin: Basin,
                        depth: Real = 0, coast: str | None = None,
                        stat: str = "Mean"):
        if variable_name not in self._variables:
            raise ValueError(
                'This {} does not contain a variable named "{}"'.format(
                    self.__class__.__name__,
                    variable_name
                )
            )
        file_path = self.path / (variable_name + '.nc')

        if coast is None:
            if len(self._costals) > 1:
                raise ValueError(
                    'coast must be specified: choose one among {}'.format(
                        ', '.join(['"' + t + '"' for t in self._costals])
                    )
                )
            coast = self._costals[0]

        if coast not in self._costals:
            raise ValueError(
                'This {} does not contain a coast value named "{}"'.format(
                    self.__class__.__name__,
                    coast
                )
            )

        with netCDF4.Dataset(file_path, 'r') as f:
            # Find basin index
            f_basins = [t.strip() for t in f.sub___list.split(',')]
            try:
                b_index = f_basins.index(basin.get_uuid())
            except ValueError:
                raise IndexError(
                    'Unable to find basin {} (UUID: {}) inside file {}'.format(
                        basin,
                        basin.get_uuid(),
                        file_path
                    )
                )

            # Find coast index
            f_coasts = [t.strip() for t in f.coast_list.split(',')]
            try:
                c_index = f_coasts.index(coast)
            except ValueError:
                raise IndexError(
                    'Unable to find coast value {} inside file {}'.format(
                        coast,
                        file_path
                    )
                )

            # Find stat index
            f_stats = [t.strip() for t in f.stat__list.split(',')]
            try:
                s_index = f_stats.index(stat)
            except ValueError:
                raise IndexError(
                    'Unable to find stat named {} inside file {}'.format(
                        stat,
                        file_path
                    )
                )

            data = f.variables[variable_name][:, b_index, c_index, :, s_index]

            times_s = np.array(f.variables['time'][:], dtype='datetime64[s]')
            times = np.array(times_s, dtype='datetime64[ns]')
            depths = np.array(f.variables['depth'][:], dtype='float32')

        data_array = xr.DataArray(
            data,
            dims=('time', 'depth'),
            coords={'time': times, 'depth': depths},
        )
        data_array_at_depth = data_array.sel(depth=depth, method='nearest')
        return TimeSeries(self.name, variable_name, basin, data_array_at_depth)
