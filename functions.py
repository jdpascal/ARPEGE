from pandas import read_excel
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import soundfile as sf

import re
import pandas as pd

POS_TO_NUM = {
    "B11": 1, "B21": 2, "B22": 3,
    "E1": 4, "E1b": 5,
    "E2": 6, "E2b": 7,
    "E3": 8, "E3b": 9,
}


import numpy as np
import matplotlib.pyplot as plt

def plot_room(df, room_value):
    # df must contain: room, pos_num, device, x, y
    d = df[df["room"] == room_value].copy()

    # keep only valid pos_num values
    d = d[d["pos_num"].notna()].copy()
    d["pos_num"] = d["pos_num"].astype(int)

    # keep only RIRs with the source facing the em32 microphone
    d = d[
        (d["orientation"] == "front")
        & (d["orientation_value"] == 32)
        & (d["type"] == "RIR")
        & (d["mic_id"] == 32)
    ]

    fig, ax = plt.subplots(figsize=(10, 10))

    # colors by position batch (pos_num)
    # C0..C8: matplotlib default palette
    pos_colors = {p: f"C{(p - 1) % 10}" for p in sorted(d["pos_num"].unique())}

    # plot points + connections
    for p in sorted(d["pos_num"].unique()):
        c = pos_colors[p]
        dp = d[d["pos_num"] == p]

        # retrieve one src, em32, and em64 point if present
        src = dp["src_xyz"].to_numpy()[0][:2]
        em32 = dp["em32_xyz"].to_numpy()[0][:2]
        em64 = dp["em64_xyz"].to_numpy()[0][:2]
        size = dp["size_room"].to_numpy()[0][:2]
        if room_value == 4 :
            corners = dp["corner_list"].to_numpy()[0]
            ax.plot(
                [corners[0][0], corners[1][0], corners[2][0], corners[3][0], corners[4][0], corners[5][0], corners[0][0]],
                [corners[0][1], corners[1][1], corners[2][1], corners[3][1], corners[4][1], corners[5][1], corners[0][1]],
                color="black",
                linestyle="-"
            )
        else :
            # plot the room as a rectangle if size_room exists
            ax.plot(
                [0, size[0], size[0], 0, 0],
                [0, 0, size[1], size[1], 0],
                color="black",
                linestyle="-"
            )

        # scatter plot points, using the same color for this pos_num
        if len(src):
            ax.scatter(src[0], src[1], color=c, marker="^", label="src")
            # ax.text(src[0], src[1], "src")

        if len(em32):
            ax.scatter(
                em32[0],
                em32[1],
                color=c,
                marker="o",
                label="em32",
                s=60,
                facecolors="none",
                edgecolors=c
            )
            # ax.text(em32[0], em32[1], "em32")

        if len(em64):
            ax.scatter(em64[0], em64[1], color=c, marker="+", label="em64", s=30)
            # ax.text(em64[0], em64[1], "em64")

        # dashed connections if src exists
        if len(src):
            sx, sy = src[0], src[1]

            x, y = em32
            ax.plot([sx, x], [sy, y], linestyle="--", color=c)

            x, y = em64
            ax.plot([sx, x], [sy, y], linestyle="--", color=c)

        # legend by pos_num, only once
        ax.plot([], [], linestyle="--", color=c, label=f"pos_num={p}")
        # ax.text(src[0], src[1], f"pos{p}", fontsize=8, color=c)

    ax.set_title(f"Room {room_value} - (src, em32, em64) per position")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()
    plt.show()

from scipy.signal import butter, sosfilt

def cartesian_to_spherical(cartesian_coords):
    x, y, z = cartesian_coords

    # Compute the radial distance
    r = np.sqrt(x**2 + y**2 + z**2)

    # Compute the azimuth angle in the XY plane
    phi = np.arctan2(y, x)  # Use np.arctan2 to avoid sign ambiguity

    rho = np.hypot(x, y)          # projection onto the XY plane
    theta = np.arctan2(z, rho)    # elevation angle
    # phi = 0.0                   # roll angle

    # Convert angles to degrees
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)

    return r, theta_deg, phi_deg


from scipy.signal import butter, sosfilt


def octave_band_sos(fc, fs, order=4):
    """
    Create a 1-octave band-pass filter centered at fc.
    Band edges: fc / sqrt(2) to fc * sqrt(2)
    """
    f1 = fc / np.sqrt(2)
    f2 = fc * np.sqrt(2)

    nyq = fs / 2
    if f2 >= nyq:
        return None

    sos = butter(order, [f1, f2], btype="bandpass", fs=fs, output="sos")
    return sos


def schroeder_decay_db(h):
    """
    Compute the Schroeder energy decay curve in dB.
    """
    h = np.asarray(h, dtype=float)
    e = h**2
    edc = np.cumsum(e[::-1])[::-1]

    # Avoid log(0)
    edc = np.maximum(edc, 1e-30)
    edc_db = 10 * np.log10(edc / np.max(edc))
    return edc_db


def fit_decay_segment(t, edc_db, db_start=-5.0, db_end=-35.0):
    """
    Fit a straight line to the [db_start, db_end] segment of the EDC curve in dB.
    Returns the slope, intercept, and the indices used for the fit.
    """
    mask = (edc_db <= db_start) & (edc_db >= db_end)

    if np.count_nonzero(mask) < 2:
        return None, None, None

    x = t[mask]
    y = edc_db[mask]

    p = np.polyfit(x, y, 1)  # y = a*x + b
    slope, intercept = p[0], p[1]
    return slope, intercept, mask


def rt_from_slope(slope, decay_range_db):
    """
    Convert a decay slope into an RT extrapolated to -60 dB.
    Example:
      - T20: range = 20 dB, from -5 to -25, RT = 3 * T20
      - T30: range = 30 dB, from -5 to -35, RT = 2 * T30
    More directly:
      RT60 = -60 / slope   (slope in dB/s, negative)
    """
    if slope is None or slope >= 0:
        return np.nan
    return -60.0 / slope


def compute_rt_band(
    rir,
    fs,
    fc,
    method="T30",
    filter_order=4,
    trim_direct_start=False,
):
    """
    Compute the reverberation time for a single octave band from a mono RIR.

    Parameters
    ----------
    rir : array-like, shape (n_samples,)
    fs : int or float
    fc : float
        Center frequency of the octave band.
    method : {'EDT', 'T05', 'T15', 'T20', 'T30'}
    filter_order : int
    trim_direct_start : bool
        If True, start the decay at the main peak.
        Useful if the RIR contains silence before the impulse.

    Returns
    -------
    result : dict
    """
    rir = np.asarray(rir, dtype=float)

    sos = octave_band_sos(fc, fs, order=filter_order)
    if sos is None:
        return {
            "fc": fc,
            "rt60": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "n_points": 0,
        }

    h = sosfilt(sos, rir)

    if trim_direct_start:
        peak_idx = np.argmax(np.abs(h))
        h = h[peak_idx:]

    edc_db = schroeder_decay_db(h)
    t = np.arange(len(h)) / fs

    if method.upper() == "EDT":
        db_start, db_end = 0.0, -10.0
    elif method.upper() == "T20":
        db_start, db_end = -5.0, -25.0
    elif method.upper() == "T30":
        db_start, db_end = -5.0, -35.0
    elif method.upper() == "T15":
        db_start, db_end = -5.0, -20.0
    elif method.upper() == "T05":
        db_start, db_end = -5.0, -10.0
    else:
        raise ValueError("method must be 'EDT', 'T05', 'T15', 'T20', or 'T30'")

    slope, intercept, mask = fit_decay_segment(
        t,
        edc_db,
        db_start=db_start,
        db_end=db_end
    )

    if slope is None:
        return {
            "fc": fc,
            "rt60": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "n_points": 0,
        }

    rt60 = rt_from_slope(slope, abs(db_end - db_start))

    return {
        "fc": fc,
        "rt60": rt60,
        "slope": slope,
        "intercept": intercept,
        "n_points": int(np.count_nonzero(mask)),
    }


def compute_rt60_octaves_multichannel(
    rir_mc,
    fs,
    center_freqs=(125, 250, 500, 1000, 2000, 4000, 8000),
    method="T30",
    filter_order=4,
    trim_direct_start=True,
    aggregate="median",
):
    """
    Compute RT60 in octave bands for a multichannel RIR.

    Parameters
    ----------
    rir_mc : ndarray, shape (n_samples, n_channels) or (n_samples,)
    fs : float
    center_freqs : iterable
    method : {'EDT', 'T05', 'T15', 'T20', 'T30'}
    aggregate : {'mean', 'median', None}
        - None: return only per-channel values
        - 'mean': average across channels
        - 'median': median across channels

    Returns
    -------
    results : dict
        {
          'per_channel': ndarray, shape (n_bands, n_channels),
          'aggregate': ndarray, shape (n_bands,) or None,
          'center_freqs': ndarray
        }
    """
    rir_mc = np.asarray(rir_mc, dtype=float)

    if rir_mc.ndim == 1:
        rir_mc = rir_mc[:, None]

    n_samples, n_channels = rir_mc.shape
    center_freqs = np.asarray(center_freqs, dtype=float)

    per_channel = np.full((len(center_freqs), n_channels), np.nan)

    for ch in range(n_channels):
        rir = rir_mc[:, ch]
        for i, fc in enumerate(center_freqs):
            res = compute_rt_band(
                rir,
                fs,
                fc,
                method=method,
                filter_order=filter_order,
                trim_direct_start=trim_direct_start,
            )
            per_channel[i, ch] = res["rt60"]

    agg = None
    if aggregate == "mean":
        agg = np.nanmean(per_channel, axis=1)
    elif aggregate == "median":
        agg = np.nanmedian(per_channel, axis=1)
    elif aggregate is None:
        agg = None
    else:
        raise ValueError("aggregate should be 'mean', 'median', or None")

    return {
        "center_freqs": center_freqs,
        "per_channel": per_channel,
        "aggregate": agg,
    }