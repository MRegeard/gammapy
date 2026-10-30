# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import html
import logging
import abc
import numpy as np
import operator
from .event_list import EventList
from .obs_table import ObservationTable

__all__ = ["ObservationFilter"]

log = logging.getLogger(__name__)


class Filter:
    def mask(self, target):
        NotImplemented

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)

    def apply(self, obj):
        if isinstance(obj, EventList):
            return obj.select_row_subset(self.mask(obj))
        elif isinstance(obj, ObservationTable):
            return obj[self.mask(obj)]
        else:
            raise TypeError("Bad type given to filter")


class And(Filter):
    def __init__(self, *filters):
        self.filters = filters

    def mask(self, obj):
        return np.logical_and.reduce([f.mask(obj) for f in self.filters])

    def __repr__(self):
        return "(" + " & ".join(map(repr, self.filters)) + ")"


class Or(Filter):
    def __init__(self, *filters):
        self.filters = filters

    def mask(self, obj):
        return np.logical_or.reduce([f.mask(obj) for f in self.filters])

    def __repr__(self):
        return "(" + " | ".join(map(repr, self.filters)) + ")"


class Not(Filter):
    def __init__(self, filter):
        self.filter = filter

    def mask(self, obj):
        return ~self.filter.mask(obj)

    def __repr__(self):
        return f"~{self.filter!r}"


class Compare(Filter):
    _symbols = {
        operator.lt: "<",
        operator.le: "<=",
        operator.gt: ">",
        operator.ge: ">=",
        operator.eq: "==",
        operator.ne: "!=",
    }

    def __init__(self, name, op, value):
        self.name = name
        self.op = op
        self.value = value

    def mask(self, obj):
        return self.op(resolve(obj, self.name), self.value)

    def __repr__(self):
        return f"{self.name} {self._symbols[self.op]} {self.value}"


def resolve(obj, name):
    value = getattr(obj, name, None)
    if value is not None and not callable(value):
        return value
    if isinstance(obj, EventList):
        if obj.table[name].unit is None:
            return obj.table[name]
        return obj.table[name].quantity
    if isinstance(obj, ObservationTable):
        if obj[name].unit is None:
            return obj[name]
        return obj[name].quantity


class Field:
    def __init__(self, name):
        self.name = name

    def __lt__(self, v):
        return Compare(self.name, operator.lt, v)

    def __le__(self, v):
        return Compare(self.name, operator.le, v)

    def __gt__(self, v):
        return Compare(self.name, operator.gt, v)

    def __ge__(self, v):
        return Compare(self.name, operator.ge, v)

    def __eq__(self, v):
        return Compare(self.name, operator.eq, v)

    def __ne__(self, v):
        return Compare(self.name, operator.ne, v)

    def between(self, lo, hi):
        return (self >= lo) & (self < hi)


class ObservationFilter:
    """Holds and applies filters to observation data.

    Parameters
    ----------
    time_filter : `astropy.time.Time`, optional
        Start and stop time of the selected time interval. Currently, we only support
        a single time interval. Default is None.
    event_filters : list of dict, optional
        An event filter dictionary needs two keys:

        - **type** : str, one of the keys in `~gammapy.data.ObservationFilter.EVENT_FILTER_TYPES`
        - **opts** : dict, it is passed on to the method of the `~gammapy.data.EventListBase`
          class that corresponds to the filter type
          (see `~gammapy.data.ObservationFilter.EVENT_FILTER_TYPES`)

        The filtered event list will be an intersection of all filters. A union
        of filters is not supported yet. Default is None.

    inverted_time : bool, optional
        Whether to invert selection i.e. to keep all entries outside the time range.
        Default is False.

    Examples
    --------
    >>> from gammapy.data import ObservationFilter, DataStore, Observation
    >>> from astropy.time import Time
    >>> from astropy.coordinates import Angle
    >>>
    >>> time_filter = Time(["2021-03-27T20:10:00", "2021-03-27T20:20:00"])
    >>> phase_filter = {
    ...     "type": "custom",
    ...     "opts": dict(parameter="PHASE", band=(0.2, 0.8)),
    ... }
    >>>
    >>> my_obs_filter = ObservationFilter(
    ...     time_filter=time_filter, event_filters=[phase_filter]
    ... )
    >>>
    >>> ds = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps")
    >>> my_obs = ds.obs(obs_id=111630)
    >>> my_obs.obs_filter = my_obs_filter
    """

    EVENT_FILTER_TYPES = dict(sky_region="select_region", custom="select_parameter")

    def __init__(self, time_filter=None, event_filters=None, inverted_time=False):
        self.time_filter = time_filter
        self.event_filters = event_filters or []
        self.inverted_time = inverted_time

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    @property
    def livetime_fraction(self):
        """Fraction of the livetime kept when applying the event_filters."""
        return self._check_filter_phase(self.event_filters)

    def filter_events(self, events):
        """Apply filters to an event list.

        Parameters
        ----------
        events : `~gammapy.data.EventListBase`
            Event list to which the filters will be applied.

        Returns
        -------
        filtered_events : `~gammapy.data.EventListBase`
            The filtered event list.
        """
        filtered_events = self._filter_by_time(events)

        for f in self.event_filters:
            method_str = self.EVENT_FILTER_TYPES[f["type"]]
            filtered_events = getattr(filtered_events, method_str)(**f["opts"])

        return filtered_events

    def filter_gti(self, gti):
        """Apply filters to a GTI table.

        Parameters
        ----------
        gti : `~gammapy.data.GTI`
            GTI table to which the filters will be applied.

        Returns
        -------
        filtered_gti : `~gammapy.data.GTI`
            The filtered GTI table.
        """
        return self._filter_by_time(gti)

    def _filter_by_time(self, data):
        """Return a new time filtered data object.

        Calls the `select_time` method of the data object.
        """
        if self.time_filter:
            return data.select_time(self.time_filter, self.inverted_time)
        else:
            return data

    def copy(self):
        """Copy the `ObservationFilter` object."""
        return copy.deepcopy(self)

    @staticmethod
    def _check_filter_phase(event_filter):
        if not event_filter:
            return 1
        for f in event_filter:
            if f.get("opts").get("parameter") == "PHASE":
                band = f.get("opts").get("band")
                return band[1] - band[0]
            else:
                return 1
