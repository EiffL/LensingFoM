#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
mkdir -p "${DATA_DIR}"

# Download and extract Gower Street sim00001
SIM_URL="http://star.ucl.ac.uk/GowerStreetSims/simulations/sim00001.tar.gz"
SIM_DIR="${DATA_DIR}/sim00001"
if [ ! -d "${SIM_DIR}" ]; then
    echo "Downloading sim00001..."
    wget -c "${SIM_URL}" -O "${DATA_DIR}/sim00001.tar.gz"
    echo "Extracting sim00001..."
    mkdir -p "${SIM_DIR}"
    tar xzf "${DATA_DIR}/sim00001.tar.gz" -C "${DATA_DIR}"
    # The tarball extracts to sim00001/sim00001/; flatten if needed
    if [ -d "${SIM_DIR}/sim00001" ]; then
        mv "${SIM_DIR}/sim00001"/* "${SIM_DIR}/"
        rm -rf "${SIM_DIR}/sim00001"
    fi
    rm "${DATA_DIR}/sim00001.tar.gz"
    echo "sim00001 extracted to ${SIM_DIR}"
else
    echo "sim00001 already exists, skipping download."
fi

# Download DES Y3 2pt data vector (MagLim, with n(z))
FITS_FILE="2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits"
FITS_PATH="${DATA_DIR}/${FITS_FILE}"
if [ ! -f "${FITS_PATH}" ]; then
    echo "Downloading DES Y3 2pt FITS file..."
    FITS_URL="https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/datavectors/${FITS_FILE}"
    wget -c "${FITS_URL}" -O "${FITS_PATH}" || {
        echo "Primary URL failed. Trying alternative..."
        FITS_URL="http://star.ucl.ac.uk/~fll/lensing/${FITS_FILE}"
        wget -c "${FITS_URL}" -O "${FITS_PATH}"
    }
    echo "DES Y3 FITS file saved to ${FITS_PATH}"
else
    echo "DES Y3 FITS file already exists, skipping download."
fi

echo "All data downloaded successfully."
