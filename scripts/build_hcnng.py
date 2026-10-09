"""TrustMR experiments: fetch and build the pinned HCNNG graph-construction code."""
import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "0fab25eb61b8852f4b4d3d4c05e0c9929647c96a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    upstream = ROOT / "third_party/hcnng"
    if not upstream.exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/jalvarm/hcnng.git", str(upstream)], check=True)
    actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        subprocess.run(["git", "-C", str(upstream), "checkout", "--detach", REVISION], check=True)
    native = ROOT / "baselines/vgann/reproduction_v2/native"
    for name in ("common.h", "hcnng.cpp"):
        if (native / name).exists():
            raise FileExistsError(f"Build source already exists: {native / name}")
    for name in ("common.h", "hcnng.cpp"):
        shutil.copyfile(upstream / name, native / name)
    subprocess.run(["patch", "-p1", "-i", str(native / "reproducibility.patch")], cwd=native, check=True)
    subprocess.run([args.cxx, "-std=c++11", "-O3", "-fopenmp", "hcnng.cpp", "-o", "hcnng"], cwd=native, check=True)
    print(native / "hcnng")


if __name__ == "__main__":
    main()
