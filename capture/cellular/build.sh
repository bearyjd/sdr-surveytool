#!/usr/bin/env bash
# Builds the vendored LTE-Cell-Scanner's CellSearch binary, hardware-free
# (no BladeRF/HackRF/OpenCL). Verified working on Fedora 43; other distros
# need equivalent packages under different names.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENDOR_DIR="$REPO_ROOT/capture/cellular/vendor/lte-cell-scanner"

if [ ! -d "$VENDOR_DIR/.git" ] && [ ! -f "$VENDOR_DIR/.git" ]; then
    echo "error: $VENDOR_DIR submodule not initialized." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

echo "== Installing system build dependencies (Fedora/dnf) =="
sudo dnf install -y rtl-sdr-devel boost-devel fftw-devel ncurses-devel \
    blas-devel lapack-devel cmake gcc-c++ make git
# rtl-sdr-devel is required even for this hardware-free build: LTE-Cell-Scanner's
# CMakeLists.txt unconditionally FIND_PACKAGE(RTLSDR REQUIRED) whenever BladeRF
# and HackRF are both disabled (verified — USE_RTLSDR is dead code, never gated
# on). No RTL-SDR hardware is touched at runtime; this is a build-time-only lib.

if ! ldconfig -p | grep -q libitpp; then
    echo "== Building IT++ from source (not packaged on Fedora) =="
    ITPP_SRC="$(mktemp -d)"
    git clone --depth 1 https://git.code.sf.net/p/itpp/git "$ITPP_SRC"
    mkdir "$ITPP_SRC/build"
    (
        cd "$ITPP_SRC/build"
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
        make -j"$(nproc)"
        sudo make install
    )
    sudo ldconfig
    rm -rf "$ITPP_SRC"
fi

echo "== Building CellSearch (hardware-free: no BladeRF/HackRF/OpenCL) =="
mkdir -p "$VENDOR_DIR/build"
(
    cd "$VENDOR_DIR/build"
    cmake -DUSE_OPENCL=0 -DUSE_BLADERF=0 -DUSE_HACKRF=0 -DCMAKE_BUILD_TYPE=Release ..
    make -j"$(nproc)"
)

echo "== Done. Binary at $VENDOR_DIR/build/src/CellSearch =="
echo "Note: IT++ installs to /usr/local, which is not on the default loader"
echo "path — set LD_LIBRARY_PATH=/usr/local/lib when running CellSearch."
