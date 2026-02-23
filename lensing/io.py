"""Data loading for Gower Street simulations and DES Y3 n(z)."""

import csv
import re
from pathlib import Path

import numpy as np
from astropy.io import fits


def load_sim_params(csv_path, sim_id=1):
    """Load cosmological parameters for a given simulation from gower_street_runs.csv.

    Parameters
    ----------
    csv_path : str or Path
        Path to gower_street_runs.csv.
    sim_id : int
        Serial number of the simulation (1-indexed).

    Returns
    -------
    dict with keys: Omega_m, sigma_8, w, Omega_bh2, h, n_s, m_nu, Omega_b
    """
    with open(csv_path) as f:
        reader = csv.reader(f)
        header1 = next(reader)  # category row
        header2 = next(reader)  # column names
        for row in reader:
            if int(row[0]) == sim_id:
                return {
                    "Omega_m": float(row[3]),
                    "sigma_8": float(row[4]),
                    "w": float(row[5]),
                    "Omega_bh2": float(row[6]),
                    "h": float(row[7]),
                    "n_s": float(row[8]),
                    "m_nu": float(row[9]),
                    "Omega_b": float(row[10]),
                }
    raise ValueError(f"Simulation {sim_id} not found in {csv_path}")


def load_shell_info(sim_dir):
    """Parse z_values.txt from a simulation directory.

    Returns a structured numpy array with columns:
        shell_id, z_far, z_near, chi_far, chi_near, z_mid, chi_mid, delta_chi
    Distances are in Mpc/h as stored in the file.
    """
    sim_dir = Path(sim_dir)
    z_file = sim_dir / "z_values.txt"

    rows = []
    with open(z_file) as f:
        header = f.readline().strip()
        # Header is CSV: # Step,z_far,z_near,delta_z,cmd_far(Mpc/h),cmd_near(Mpc/h),...
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split(",")
            rows.append([float(v) for v in vals])

    data = np.array(rows)

    # Columns: Step(0), z_far(1), z_near(2), delta_z(3),
    #          cmd_far[Mpc/h](4), cmd_near[Mpc/h](5), delta_cmd[Mpc/h](6),
    #          cmd/box_far(7), cmd/box_near(8), delta_cmd/box(9)
    shell_ids = data[:, 0].astype(int)
    z_far = data[:, 1]
    z_near = data[:, 2]
    chi_far = data[:, 4]  # Mpc/h
    chi_near = data[:, 5]  # Mpc/h

    z_mid = 0.5 * (z_far + z_near)
    chi_mid = 0.5 * (chi_far + chi_near)
    delta_chi = data[:, 6]  # Mpc/h, directly from the file

    dt = np.dtype([
        ("shell_id", int),
        ("z_far", float),
        ("z_near", float),
        ("chi_far", float),
        ("chi_near", float),
        ("z_mid", float),
        ("chi_mid", float),
        ("delta_chi", float),
    ])
    result = np.zeros(len(shell_ids), dtype=dt)
    result["shell_id"] = shell_ids
    result["z_far"] = z_far
    result["z_near"] = z_near
    result["chi_far"] = chi_far
    result["chi_near"] = chi_near
    result["z_mid"] = z_mid
    result["chi_mid"] = chi_mid
    result["delta_chi"] = delta_chi

    return result


def load_shell_map(sim_dir, shell_id):
    """Load a single lightcone shell HEALPix map (particle counts per pixel).

    Parameters
    ----------
    sim_dir : str or Path
    shell_id : int

    Returns
    -------
    np.ndarray : HEALPix map of particle counts (nside=2048, RING ordering)
    """
    sim_dir = Path(sim_dir)
    fname = sim_dir / f"run.{shell_id:05d}.lightcone.npy"
    return np.load(fname)


def get_valid_shell_ids(sim_dir):
    """List valid (non-incomplete) shell IDs in a simulation directory.

    Returns sorted list of integer shell IDs for which
    run.XXXXX.lightcone.npy exists (excluding .incomplete.npy files).
    """
    sim_dir = Path(sim_dir)

    # Get all incomplete shell IDs
    incomplete = set()
    for f in sim_dir.glob("run.*.incomplete.npy"):
        m = re.search(r"run\.(\d+)\.incomplete\.npy", f.name)
        if m:
            incomplete.add(int(m.group(1)))

    # Get all lightcone shell IDs, excluding incomplete ones
    valid = []
    for f in sim_dir.glob("run.*.lightcone.npy"):
        m = re.search(r"run\.(\d+)\.lightcone\.npy", f.name)
        if m:
            sid = int(m.group(1))
            if sid not in incomplete:
                valid.append(sid)

    return sorted(valid)


def load_des_y3_nz(fits_path):
    """Load DES Y3 MagLim n(z) from the 2pt data vector FITS file.

    Parameters
    ----------
    fits_path : str or Path

    Returns
    -------
    z_mid : np.ndarray, shape (n_z,)
        Midpoints of redshift bins.
    nz_bins : np.ndarray, shape (4, n_z)
        Normalised n(z) for each of the 4 tomographic bins.
    """
    with fits.open(fits_path) as hdul:
        nz_ext = hdul["NZ_SOURCE"]
        z_low = nz_ext.data["Z_LOW"]
        z_mid = nz_ext.data["Z_MID"]
        z_high = nz_ext.data["Z_HIGH"]

        nz_bins = np.zeros((4, len(z_mid)))
        for i in range(4):
            nz_bins[i] = nz_ext.data[f"BIN{i+1}"]

    # Normalise each bin so that integral n(z) dz = 1
    dz = z_high - z_low
    for i in range(4):
        norm = np.trapezoid(nz_bins[i], z_mid)
        if norm > 0:
            nz_bins[i] /= norm

    return z_mid, nz_bins
