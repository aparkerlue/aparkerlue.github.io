# -*- mode: python; coding: utf-8; -*-
from functools import reduce
import gzip
import io
from itertools import permutations
import logging
import operator
import os
import re
import shutil

from bokeh.io import output_file, show
from bokeh.models import (
    ColumnDataSource,
    GeoJSONDataSource,
    GMapOptions,
    GMapPlot,
    Range1d,
    PanTool,
    BoxZoomTool,
    BoxSelectTool,
    WheelZoomTool,
    ResetTool,
    SaveTool,
    ZoomInTool,
    ZoomOutTool,
)
from bokeh.models.glyphs import Circle, Patches, Segment, Triangle
from bokeh.plotting import figure
from ediblepickle import checkpoint
from geopy.distance import distance
import networkx as nx
import numpy as np
import pandas as pd
import requests
from scipy.integrate import dblquad, simps
from scipy.stats import multivariate_normal
from shapely.geometry import shape, Point
from shapely.ops import unary_union
import ujson as json
from tqdm import tqdm

from settings import (
    TAXI_DATA_DIR,
    GOOGLE_API_KEY,
    GEOJSON_FILE_MANHATTAN,
    GEOJSON_FILE_NYC_NEIGHBORHOODS,
)

DATA_TYPE_PARAMS = {
    "yellow": {
        "dtype": {
            "VendorID": np.int_,
            "passenger_count": np.int_,
            "trip_distance": np.float_,
            "pickup_longitude": np.float_,
            "pickup_latitude": np.float_,
            "RatecodeID": np.int_,
            "store_and_fwd_flag": object,
            "dropoff_longitude": np.float_,
            "dropoff_latitude": np.float_,
            "payment_type": np.int_,
            "fare_amount": np.float_,
            "extra": np.float_,
            "mta_tax": np.float_,
            "tip_amount": np.float_,
            "tolls_amount": np.float_,
            "improvement_surcharge": np.float_,
            "total_amount": np.float_,
        },
        "parse_dates": [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
        ],
        "sort_key": "tpep_pickup_datetime",
    },
    "green": {
        "dtype": {
            "VendorID": np.int_,
            "Store_and_fwd_flag": object,
            "RateCodeID": np.int_,
            "Pickup_longitude": np.float_,
            "Pickup_latitude": np.float_,
            "Dropoff_longitude": np.float_,
            "Dropoff_latitude": np.float_,
            "Passenger_count": np.int_,
            "Trip_distance": np.float_,
            "Fare_amount": np.float_,
            "Extra": np.float_,
            "MTA_tax": np.float_,
            "Tip_amount": np.float_,
            "Tolls_amount": np.float_,
            "Ehail_fee": np.float_,
            "improvement_surcharge": np.float_,
            "Total_amount": np.float_,
            "Payment_type": np.int_,
            "Trip_type": np.int_,
        },
        "parse_dates": [
            "lpep_pickup_datetime",
            "Lpep_dropoff_datetime",
        ],
        "sort_key": "lpep_pickup_datetime",
    },
}


class TaxiData:

    PICKUP_TIME_FIELD = {
        "yellow": "tpep_pickup_datetime",
        "green": "lpep_pickup_datetime",
        "self": "pick_time",
    }
    SELECT_FIELDS = {
        "yellow": [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
            "passenger_count",
        ],
        "green": [
            "lpep_pickup_datetime",
            "Lpep_dropoff_datetime",
            "Pickup_latitude",
            "Pickup_longitude",
            "Dropoff_latitude",
            "Dropoff_longitude",
            "Passenger_count",
        ],
        "self": [
            "pick_time",
            "drop_time",
            "pick_lat",
            "pick_lon",
            "drop_lat",
            "drop_lon",
            "n_passengers",
        ],
    }

    def __init__(self, *data_files, data_dir=TAXI_DATA_DIR, datasets=None):
        self.data_files = data_files
        self.data_dir = data_dir

        self.datasets = datasets

    def load(self):
        """Load data sets.

        Sort data by pick-up time. Do not reload if already loaded.

        """
        if self.datasets:
            return

        dataset_types = []
        for f in self.data_files:
            if re.match("yellow_", f):
                dataset_types.append("yellow")
            elif re.match("green_", f):
                dataset_types.append("green")
            else:
                raise ValueError(
                    "Can't infer data set type from filename `{}'".format(f)
                )

        self.datasets = []
        for f, t in zip(self.data_files, dataset_types):
            fpath = os.path.join(self.data_dir, f)
            x = taxi_read_sort_tripdata(fpath, t)
            self.datasets.append({"type": t, "data": x})

    def extract_pickup_window(self, start_time, duration):
        """Return taxi data inside the specified pick-up window.

        Specify `duration` in minutes. This function returns a copy of
        the data.

        """
        self.load()

        start_time = pd.Timestamp(start_time)
        end_time = start_time + pd.Timedelta(minutes=duration)
        bvectors = []
        for x in self.datasets:
            pickup_field, df = self.PICKUP_TIME_FIELD[x["type"]], x["data"]
            bvectors.append(
                (df[pickup_field] >= start_time) & (df[pickup_field] < end_time)
            )

        df = self._merge_data_subsets(bvectors)
        df.sort_values(self.PICKUP_TIME_FIELD["self"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _merge_data_subsets(self, bvectors):
        """Return merged data sets, filtered on boolean vectors.

        This function produces a copy of the data.

        """
        assert len(bvectors) == len(self.datasets)
        frames = []
        for i, x in zip(bvectors, self.datasets):
            typ, df = x["type"], x["data"]
            field_map = {
                old: new
                for old, new in zip(self.SELECT_FIELDS[typ], self.SELECT_FIELDS["self"])
            }
            frames.append(df[i][self.SELECT_FIELDS[typ]].rename(columns=field_map))
        return pd.concat(frames, ignore_index=True)


def fetch_from_url(url, destination_dir=".", use_gzip=False):
    """Fetch file from URL and return path of fetched file."""
    fpath = os.path.join(destination_dir, url.split("/")[-1])
    if use_gzip:
        fpath += ".gz"
        open_fn = gzip.open
    else:
        open_fn = open
    with requests.get(url, stream=True) as r:
        with open_fn(fpath, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    return fpath


def fetch_taxi_data(url_file, **params):
    """Fetch taxi data using file of URLs."""
    with open(url_file) as f:
        L = [x.strip() for x in f]
    for url in tqdm(L):
        logging.info("Fetching {}...".format(url))
        fetch_from_url(url, **params)


def pickle_pd_to_csv(x, f):
    """Write dataframe `x` to `BufferedWriter` `f` as gzipped CSV."""
    with gzip.GzipFile(fileobj=f) as g, io.TextIOWrapper(g) as w:
        x.to_csv(w, index=False)


def unpickle_csv_to_pd(f):
    """Read dataframe from `BufferedReader` `f` to gzipped CSV."""
    data_type = "yellow" if re.search("_yellow\..*", f.name) else "green"
    dtype = DATA_TYPE_PARAMS[data_type]["dtype"]
    parse_dates = DATA_TYPE_PARAMS[data_type]["parse_dates"]
    with gzip.GzipFile(fileobj=f) as g, io.TextIOWrapper(g) as w:
        df = pd.read_csv(w, dtype=dtype, parse_dates=parse_dates)
    return df


@checkpoint(
    key=lambda args, kwargs: "taxi_read_sort_tripdata_{}_{}.csv.gz".format(
        os.path.splitext(os.path.splitext(os.path.basename(args[0]))[0])[0],
        args[1],
    ),
    pickler=pickle_pd_to_csv,
    unpickler=unpickle_csv_to_pd,
    work_dir="checkpoints",
)
def taxi_read_sort_tripdata(csvfile, data_type):
    """Read taxi data to a Pandas dataframe and sort by pickup time."""
    if data_type not in {"yellow", "green"}:
        raise ValueError("`data_type` must be either yellow or green")

    dtype = DATA_TYPE_PARAMS[data_type]["dtype"]
    parse_dates = DATA_TYPE_PARAMS[data_type]["parse_dates"]
    sort_key = DATA_TYPE_PARAMS[data_type]["sort_key"]

    df = pd.read_csv(csvfile, dtype=dtype, parse_dates=parse_dates)
    df.sort_values(sort_key, inplace=True)

    return df


def point_in_zone(zone, p):
    """Return True if point `p` is inside zone `zone`.

    Examples:
    - point_in_zone('Manhattan', Point(-74, 40.73))
    - point_in_zone('Astoria', Point(-73.92, 40.76))
    """
    polygon = load_geojson(zone, make_shape=True)
    return polygon.contains(p)


def filter_trips_by_zones(df, trip_zones, include_permutations=False, index_only=False):
    """Return trips originating and terminating in specified zones.

    - `trip_zone`: Iterable of start- and end-zone pairs
    - `include_permutations`: Include zone combinations
    """
    logging.info("Initiating filter for trips by zones...")
    if isinstance(trip_zones, str):
        trip_zones = [trip_zones]

    uniq_zones = set()
    for x in trip_zones:
        if isinstance(x, str):
            uniq_zones.add(x)
        elif isinstance(x, (set, frozenset)):
            uniq_zones.add(frozenset(x))
        else:
            uniq_zones.update(x)

    # Load GeoJSON
    zone_data = {
        z: (
            load_geojson(z, make_shape=True)
            if isinstance(z, str)
            else unary_union([load_geojson(x, make_shape=True) for x in z])
        )
        for z in uniq_zones
    }

    if include_permutations:
        trip_zones = list(uniq_zones) + list(permutations(uniq_zones, r=2))

    logging.info(
        "Filtering trips for {:,} zone combinations...".format(len(trip_zones))
    )
    filters = []
    for t in trip_zones:
        if isinstance(t, str) or isinstance(t, (set, frozenset)):
            i = _filter_trips_by_zone(zone_data, df, t, index_only=True)
        else:
            i = _filter_trips_by_zone(zone_data, df, *t, index_only=True)
        filters.append(i)

    i = reduce(operator.or_, filters)
    logging.info("Filtered {:,} trips...".format(sum(i)))
    return i if index_only else df[i]


def _filter_trips_by_zone(zone_data, df, start_zone, end_zone=None, index_only=False):
    """Return trips originating and terminating in specified zones."""
    if end_zone:
        logging.info("Filtering trips from {} to {}...".format(start_zone, end_zone))
    else:
        logging.info("Filtering trips within {}...".format(start_zone))

    z_orig = zone_data[start_zone]
    z_dest = zone_data[end_zone] if end_zone else z_orig

    i = df.apply(
        lambda r: (
            z_orig.contains(Point(r["pick_lon"], r["pick_lat"]))
            and z_dest.contains(Point(r["drop_lon"], r["drop_lat"]))
        ),
        axis=1,
    )
    return i if index_only else df[i]


def load_geojson(zone, make_shape=False):
    """Return GeoJSON data for `zone`.

    Zone is either 'Manhattan' or a NYC neighborhood name. Return a
    `FeatureCollection` or `GeometryCollection` unless `make_shape` is
    True, in which case return a Shapely shape.

    """
    logging.info("Loading GeoJSON for `{}'...".format(zone))
    if zone == "Manhattan":
        with open(GEOJSON_FILE_MANHATTAN) as f:
            js = json.load(f)
        if make_shape:
            return shape(js["geometries"][0])
        else:
            return js

    # The zone must be in the NYC neighborhood data
    with open(GEOJSON_FILE_NYC_NEIGHBORHOODS) as f:
        js = json.load(f)
    if zone == "NYC":
        if make_shape:
            raise ValueError(
                """
Can't return NYC neighborhood geometry outside of FeatureCollection
            """.strip()
            )
        else:
            return js
    elif isinstance(zone, set):
        if make_shape:
            return unary_union([load_geojson(z, make_shape=True) for z in zone])
        else:
            x = [f for f in js["features"] if f["properties"]["neighborhood"] in zone]
            if not x:
                raise ValueError("Unrecognized zone `{}'".format(zone))
            js["features"] = x
            return js
    else:
        if make_shape:
            for f in js["features"]:
                if f["properties"]["neighborhood"] == zone:
                    return shape(f["geometry"])
            raise ValueError("Unrecognized zone `{}'".format(zone))
        else:
            x = [f for f in js["features"] if f["properties"]["neighborhood"] == zone]
            if not x:
                raise ValueError("Unrecognized zone `{}'".format(zone))
            js["features"] = x
            return js


def get_manhattan_neighborhoods():
    """Get list of Manhattan neighborhoods."""
    nyc = load_geojson("NYC")
    mhoods = set(
        x["properties"]["neighborhood"]
        for x in nyc["features"]
        if x["properties"]["borough"] == "Manhattan"
    ) - {
        "Roosevelt Island",
        "Randall's Island",
        "Liberty Island",
        "Marble Hill",
        "Governors Island",
        "Ellis Island",
    }
    return mhoods


def marker_angle(df, offset=-0.5 * np.pi):
    """Calculate the marker angle of for each trip in `df`."""
    return (
        df.apply(
            lambda x: (
                np.arctan2(
                    (1 if x["drop_lat"] - x["pick_lat"] > 0 else -1)
                    * distance(
                        (x["pick_lat"], x["pick_lon"]), (x["drop_lat"], x["pick_lon"])
                    ).km,
                    (1 if x["drop_lon"] - x["pick_lon"] > 0 else -1)
                    * distance(
                        (x["pick_lat"], x["pick_lon"]), (x["pick_lat"], x["drop_lon"])
                    ).km,
                )
            ),
            axis=1,
        ).values
        + offset
    )


def marker_angle2(df, offset=-0.5 * np.pi):
    """Calculate the marker angle of for each trip in `df`.

    This implementation is slower than `marker_angle`, but it is
    clearer.

    """
    dist = df.apply(
        lambda x: pd.Series(
            {
                "y": distance(
                    (x["pick_lat"], x["pick_lon"]), (x["drop_lat"], x["pick_lon"])
                ).km,
                "x": distance(
                    (x["pick_lat"], x["pick_lon"]), (x["pick_lat"], x["drop_lon"])
                ).km,
            }
        ),
        axis=1,
    )
    return (
        np.arctan2(
            np.where(df["drop_lat"] - df["pick_lat"] > 0, 1, -1) * dist["y"],
            np.where(df["drop_lon"] - df["pick_lon"] > 0, 1, -1) * dist["x"],
        ).values
        + offset
    )


def plot_geojson(zones, trips=None, plotfile=None, **figure_options):
    """Plot GeoJSON data.

    Examples:
    - plot_geojson('Manhattan')
    - plot_geojson('NYC')
    """
    p = figure(
        active_scroll="wheel_zoom",
        **figure_options,
    )
    p.toolbar.logo = None
    p.toolbar_location = "above"
    p.title.text = None

    for zone in zones:
        g = load_geojson(zone)
        geo_source = GeoJSONDataSource(geojson=json.dumps(g))
        p.patches(xs="xs", ys="ys", alpha=0.5, source=geo_source)

    if trips is not None:
        add_trips_to_plot(p, trips)

    if plotfile:
        output_file(plotfile)

    show(p)


def add_trips_to_plot(p, trips):
    """Add trip glyphs to plot `p`."""
    data_source = ColumnDataSource(
        pd.concat(
            [
                trips,
                pd.DataFrame(
                    {
                        "marker_size": 8 * trips["n_passengers"],
                        "marker_angle": marker_angle(trips),
                    }
                ),
            ],
            axis=1,
        )
    )
    p.add_glyph(
        data_source,
        Segment(
            x0="pick_lon",
            y0="pick_lat",
            x1="drop_lon",
            y1="drop_lat",
            line_color="black",
            line_width=1,
        ),
    )
    p.add_glyph(
        data_source,
        Circle(
            x="pick_lon",
            y="pick_lat",
            size="marker_size",
            fill_color="blue",
            fill_alpha=0.6,
        ),
    )
    p.add_glyph(
        data_source,
        Triangle(
            x="drop_lon",
            y="drop_lat",
            size="marker_size",
            angle="marker_angle",
            fill_color="red",
            fill_alpha=0.6,
        ),
    )


def plot_zones_and_trips(zones, trips, include_permutations=False, plotfile=None):
    """Plot trips in specified zones."""
    hoods = uniq_hoods_in_zones(zones)

    aspectratio = 4 / 3
    lo = (40.69, -74.05)
    lon_max = -73.85
    lat_max = lo[0] + (
        distance(lo, (lo[0], lo[1] + 1)).km
        / distance(lo, (lo[0] + 1, lo[1])).km
        * (lon_max - lo[1])
        / aspectratio
    )
    hi = (lat_max, lon_max)
    plot_geojson(
        hoods,
        trips=trips,
        plotfile=plotfile,
        x_range=Range1d(lo[1], hi[1]),
        y_range=Range1d(lo[0], hi[0]),
        plot_width=700,
        plot_height=int(700 / aspectratio),
    )


def calculate_centroid_mean(shapes):
    """Calculate the mean of the centroids of `shapes."""
    return np.mean(
        [(s.centroid.x, s.centroid.y) for s in shapes],
        axis=0,
    )


def plot_gmap(zones, trips=None, **map_options):
    """Plot zones over a Google Map.

    Examples:
    - plot_gmap(['Astoria', 'Manhattan'], zoom=12)
    - plot_gmap(['Astoria', 'Midtown', 'Greenpoint', 'Sunnyside', 'Harlem'])
    """

    # Gather zone data
    polygons = [GeoJSONDataSource(geojson=json.dumps(load_geojson(z))) for z in zones]
    u = unary_union([load_geojson(z, make_shape=True) for z in zones])
    m_polygons = u.centroid

    plot = GMapPlot(
        api_key=GOOGLE_API_KEY,
        x_range=Range1d(),
        y_range=Range1d(),
        map_options=GMapOptions(
            lat=m_polygons.y,
            lng=m_polygons.x,
            map_type="roadmap",
            **map_options,
        ),
    )
    plot.toolbar.logo = None
    plot.toolbar_location = "above"
    plot.title.text = None

    # Toolbar
    wheel_zoom = WheelZoomTool()
    plot.add_tools(
        PanTool(),
        BoxZoomTool(),
        BoxSelectTool(),
        wheel_zoom,
        ZoomInTool(),
        ZoomOutTool(),
        ResetTool(),
        SaveTool(),
    )
    plot.toolbar.active_scroll = wheel_zoom

    # Add neighborhood polygons
    for geo_source in polygons:
        glyph = Patches(xs="xs", ys="ys", fill_color="Blue", fill_alpha=0.2)
        plot.add_glyph(geo_source, glyph)

    # Add trips
    if trips is not None:
        add_trips_to_plot(plot, trips)

    output_file("plots/plot.html")
    show(plot)

    return plot


def make_path_compatible(s):
    """Make the string compatible with filename paths."""
    return "".join(
        c if c.isalpha() or c.isdigit() or c in "-./_" else "_" for c in s if c != ":"
    )


@checkpoint(
    key=lambda args, kwargs: "taxi_load_trip_window_{}_{}.dat".format(
        make_path_compatible(args[0]),
        args[1],
    ),
    work_dir="checkpoints",
)
def load_trip_window(start_time, duration):
    """Convenience function for loading trip data."""
    logging.info("Loading data from trip window...")
    taxidata = TaxiData(
        "yellow_tripdata_2016-06.csv.gz",
        "green_tripdata_2016-06.csv.gz",
    )
    return taxidata.extract_pickup_window(start_time, duration)


def uniq_hoods_in_zones(zones):
    """Return unique hoods in zones.

    `zones` is an iterable whose elements could be:
    - `frozenset`
    - `set`
    - `str`
    - Length-2 `tuple`

    For example:
    zones = {
        ('Astoria', 'LaGuardia Airport'),
        ('LaGuardia Airport', 'Astoria'),
        (frozenset({'Midtown', 'Kips Bay'}), 'LaGuardia Airport'),
        ('LaGuardia Airport', frozenset({'Midtown', 'Kips Bay'})),
    }
    """
    hoods = set()
    for x in zones:
        if isinstance(x, str):
            hoods.add(x)
        elif isinstance(x, (set, frozenset)):
            hoods.update([h for h in x])
        else:
            # x must be a tuple
            assert isinstance(x, tuple) and len(x) == 2
            for y in x:
                if isinstance(y, str):
                    hoods.add(y)
                elif isinstance(y, (set, frozenset)):
                    hoods.update([h for h in y])
                else:
                    raise TypeError("Unexpected type {}".format(type(y)))
    return hoods


class TruncBiNorm:
    """Truncated bivariate normal distribution.

    >>> round(TruncBiNorm(a=[-1, -1], b=[1, 1]).cdf([0, 0]), 2)
    0.25
    """

    def __init__(self, mean=[0, 0], cov=1, a=[-3, -3], b=[3, 3]):
        self.mean = np.array(mean)
        self.cov = cov
        self.a = np.array(a)  # Lower bound
        self.b = np.array(b)  # Upper bound

        self.X = multivariate_normal(self.mean, self.cov)
        self.c = self.normalizing_constant()

    def normalizing_constant(self):
        return 1 / (
            self.X.cdf(self.b)
            - self.X.cdf([self.a[0], self.b[1]])
            - self.X.cdf([self.b[0], self.a[1]])
            + self.X.cdf(self.a)
        )

    def pdf(self, x):
        if any(x < self.a) or any(x > self.b):
            return 0
        return self.c * self.X.pdf(x)

    def cdf(self, x):
        if any(x < self.a):
            return 0
        x = np.minimum(x, self.b)
        return self.c * (
            self.X.cdf(x)
            - self.X.cdf([self.a[0], x[1]])
            - self.X.cdf([x[0], self.a[1]])
            + self.X.cdf(self.a)
        )

    def recenter(self, mean):
        """Recenter the distribution."""
        d = mean - self.mean
        self.mean = np.array(mean)
        self.a += d
        self.b += d
        self.X = multivariate_normal(self.mean, self.cov)
        return self

    def __and__(self, other):
        """Binary `&` operation that returns an overlap object."""
        return TruncBiNormOverlap(self, other)


class TruncBiNormOverlap:
    """Overlap between two truncated bivariate normal distributions.

    >>> round(
    ...     (
    ...         TruncBiNorm(a=[-1, -1], b=[1, 1])
    ...         & TruncBiNorm(a=[-1, -1], b=[1, 1]).recenter([1, 0])
    ...     ).distribution_overlap(),
    ...     4,
    ... )
    0.4391
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self.a = np.maximum(X.a, Y.a)
        self.b = np.minimum(X.b, Y.b)

    def distribution_overlap(self, epsabs=1.49e-8):
        """Return overlap of two truncated bivariate normal distributions.

        The overlap is a number between 0 and 1: 0 means that the
        distribution supports are disjoint, and 1 means that the
        distributions completely overlap.

        """
        if any(self.a >= self.b):
            return 0
        y, _ = dblquad(
            lambda y, x: min(self.X.pdf([x, y]), self.Y.pdf([x, y])),
            a=self.a[0],
            b=self.b[0],
            gfun=lambda x: self.a[1],
            hfun=lambda x: self.b[1],
            epsabs=epsabs,
        )
        return y


class Trip:
    M_PER_LON = 250 / 2.96e-3  # Meters per degree longitude
    M_PER_LAT = 250 / 2.25e-3  # Meters per degree latitude

    def __init__(self, pds, velocity=250, agg_sd=300, trunc_limit=3):
        """Create a trip object from Pandas Series `pds`.

        Parameters:

        - `velocity`: Velocity stated in meters per minute

        - `agg_sd`: One standard deviation of aggregation decay in
          meters

        - `trunc_limit`: Bivariate Gaussian truncation limit in
          standard deviations

        """
        self.start_time = pds["pick_time"]
        self.orig = np.array([pds["pick_lon"], pds["pick_lat"]])
        self.dest = np.array([pds["drop_lon"], pds["drop_lat"]])
        self.n_passengers = pds["n_passengers"]

        self.velocity = velocity
        self.agg_sd = agg_sd
        self.trunc_limit = trunc_limit

    def __hash__(self):
        return hash(
            "-".join(
                map(
                    str,
                    [
                        self.start_time,
                        self.orig,
                        self.dest,
                        self.n_passengers,
                    ],
                )
            )
        )

    def distance(self):
        """Return trip distance in meters."""
        return (
            (self.M_PER_LON * (self.dest[0] - self.orig[0])) ** 2
            + (self.M_PER_LAT * (self.dest[1] - self.orig[1])) ** 2
        ) ** 0.5

    def duration(self):
        """Return trip duration in minutes."""
        return self.distance() / self.velocity

    def get_end_time(self):
        """Return the ending time of the trip."""
        return (self.start_time + pd.Timedelta(minutes=self.duration())).floor("s")

    def position(self, t):
        """Return position coordinates at time `t`.

        - `t`: Either a number (of minutes into the ride) or a Pandas
          Timestamp object.

        """
        if isinstance(t, pd.Timestamp):
            t = (t - self.start_time) / pd.Timedelta(minutes=1)

        if t < 0 or t > self.duration():
            raise ValueError("Trip undefined at time `t'")

        v = self._velocity_vector()
        x = self.orig + 1 / np.array([self.M_PER_LON, self.M_PER_LAT]) * v * t

        return x

    def _velocity_vector(self):
        """Return velocity vector in meters per minute."""
        d = self.distance()
        v = self.velocity * np.array(
            [
                self.M_PER_LON * (self.dest[0] - self.orig[0]) / d,
                self.M_PER_LAT * (self.dest[1] - self.orig[1]) / d,
            ]
        )
        assert np.allclose(np.sum(v ** 2) ** 0.5, self.velocity)
        assert np.allclose(
            self.orig
            + 1 / np.array([self.M_PER_LON, self.M_PER_LAT]) * v * self.duration(),
            self.dest,
        )
        return v

    def aggregation_efficiency(self, other, epsabs=1e-2):
        """Return efficiency of aggregating with `other` trip.

        Trip aggregation efficiency is measured as the number of
        minutes of approximate overlap between two trips.

        - If two trips completely overlap, then the aggregation
          efficiency is the total duration (in minutes) of the two
          trips.

        - If one trip is completely contained within another, then the
          efficiency is the total duration (in minutes) of the
          contained trip.

        - If two trips are disjoint, then the efficiency is zero.

        This operation is commutative.

        """
        if self.trunc_limit < 0 or other.trunc_limit < 0:
            raise ValueError("`self.trunc_limit' must be nonnegative")

        # One standard deviation longitude and latitude
        s1 = self.agg_sd / np.array([self.M_PER_LON, self.M_PER_LAT])
        s2 = other.agg_sd / np.array([other.M_PER_LON, other.M_PER_LAT])

        # Find intersection of time windows
        t0 = max(self.start_time, other.start_time)
        T = min(self.get_end_time(), other.get_end_time())
        if t0 >= T:
            return 0

        # Calculate even number of approx. one-minute periods
        periods = int(2 * np.ceil(0.5 * (T - t0) / pd.Timedelta(minutes=1)))
        t_span, m = self._timestamp_range(t0, T, periods)

        X = TruncBiNorm([0, 0], s1 ** 2, -self.trunc_limit * s1, self.trunc_limit * s1)
        Y = TruncBiNorm(
            [0, 0], s2 ** 2, -other.trunc_limit * s2, other.trunc_limit * s2
        )
        o = [
            (
                X.recenter(self.position(t)) & Y.recenter(other.position(t))
            ).distribution_overlap(epsabs=epsabs)
            for t in t_span
        ]
        return simps(o, m)

    @staticmethod
    def _timestamp_range(start, end, periods):
        """Return a range of timestamps."""
        minutes = (end - start) / pd.Timedelta(minutes=1)
        minute_span = np.linspace(0, minutes, periods + 1)
        r = np.array([start + pd.Timedelta(minutes=x) for x in minute_span])
        assert r[-1] == end
        return r, minute_span

    def cossimil(self, other):
        """Calculate cosine similarity between two trips."""
        a = self.dest - self.orig
        b = other.dest - other.orig
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if np.any(np.isclose([norm_a, norm_b], 0)):
            return 0
        return a @ b / norm_a / norm_b


def zone_aggeffs(
    trips, sample_size=None, cossimil_thresh=np.cos(np.pi / 6), use_tqdm=False
):
    """Calculate zone-level ride aggregation efficiencies."""

    def _trip_aggeff(u):
        """Calculate trip aggregation efficiency for a trip `u`."""
        T = u.duration()
        if np.isclose(T, 0):
            return 0

        for v in g.nodes:
            if v is u or g.has_edge(u, v):
                continue
            if u.cossimil(v) > cossimil_thresh:
                ae = u.aggregation_efficiency(v)
            else:
                ae = 0
            g.add_edge(u, v, ae=ae)

        tae = 0
        for _, eattr in g.adj[u].items():
            tae += eattr["ae"]
        return tae / T

    g = nx.Graph()
    g.add_nodes_from(Trip(s) for _, s in trips.iterrows())

    if sample_size and sample_size < len(g.nodes):
        trip_sample = np.random.choice(
            g.nodes,
            sample_size,
            replace=False,
        ).tolist()
    else:
        trip_sample = list(g.nodes)

    logging.info(
        "Estimating agg. efficiency for {} samples over {} trips..".format(
            len(trip_sample), len(g.nodes)
        )
    )
    tripgen = tqdm(trip_sample) if use_tqdm else trip_sample
    zae = [_trip_aggeff(u) for u in tripgen]

    return np.array(zae)
