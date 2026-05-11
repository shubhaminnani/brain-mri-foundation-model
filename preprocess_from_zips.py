"""
Preprocess FOMO300K directly from zip files → .npy (no extraction step).

Input layout:
  fomo/
    PT001_ClevelandCCF/
      sub-01/
        ses-01.zip          ← contains ses-01/anat/sub-01_ses-01_T1w.nii.gz
        ses-02.zip
      sub-02/
        ...
    PT002_Nigerian_Clinical/
      ...

Output layout:
  fomo_preprocessed/
    PT001_ClevelandCCF_sub-01_ses-01_sub-01_ses-01_T1w.npy
    ...

Preprocessing (matches baseline-codebase / fomo25):
  - Resample to 1 mm isotropic
  - Reorient to RAS
  - Crop to non-zero bounding box
  - Volume-wise z-normalisation
  - Save as float16 (halves disk vs float32)

Storage estimate: ~1.5–2.5 TB for 300 K scans.

Usage:
  python preprocess_from_zips.py \
      --in_path  /mnt/f/challenge/fomo \
      --out_path /mnt/f/challenge/fomo_preprocessed \
      --num_workers 16
"""

import argparse
import gzip
import io
import os
import tempfile
import zipfile
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from yucca.functional.preprocessing import preprocess_case_for_training_without_label
    from yucca.functional.utils.loading import read_file_to_nifti_or_np
    HAS_YUCCA = True
except ImportError:
    HAS_YUCCA = False

try:
    from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
except ImportError:
    def maybe_mkdir_p(p): os.makedirs(p, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _list_niftis_in_zip(zip_path: str) -> list:
    """Return all .nii.gz paths inside a zip."""
    with zipfile.ZipFile(zip_path) as zf:
        return sorted(n for n in zf.namelist() if n.endswith(".nii.gz"))


def _load_nifti_from_zip(zip_path: str, nii_name: str) -> nib.Nifti1Image:
    """Read a specific .nii.gz inside a zip without extracting to disk."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(nii_name) as f:
            raw = f.read()
    decompressed = gzip.decompress(raw)
    fh = nib.FileHolder(fileobj=io.BytesIO(decompressed))
    return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})


def _simple_preprocess(img: nib.Nifti1Image) -> np.ndarray:
    """
    Fallback preprocessing when yucca is not available:
      - reorient to RAS
      - volume-wise z-norm
      - crop to non-zero bounding box
    Does NOT resample to 1mm (fast but resolution-inconsistent).
    """
    img = nib.as_closest_canonical(img)
    vol = img.get_fdata(dtype=np.float32)

    # z-normalise
    mask = vol != 0
    if mask.sum() > 0:
        mu, sigma = vol[mask].mean(), vol[mask].std()
        vol = (vol - mu) / (sigma + 1e-8)
    else:
        vol = vol - vol.mean()

    # crop to non-zero
    idx = np.where(mask)
    if len(idx[0]) == 0:
        return vol
    sl = tuple(slice(ax.min(), ax.max() + 1) for ax in idx)
    return vol[sl]


def _yucca_preprocess(img: nib.Nifti1Image) -> np.ndarray:
    """Full yucca preprocessing: 1mm iso + RAS + znorm + crop."""
    images, _ = preprocess_case_for_training_without_label(
        images=[img],
        normalization_operation=["volume_wise_znorm"],
        crop_to_nonzero=True,
        target_orientation="RAS",
        target_spacing=[1.0, 1.0, 1.0],
        keep_aspect_ratio_when_using_target_size=False,
        transpose=[0, 1, 2],
    )
    return images[0]


# ── per-scan worker ───────────────────────────────────────────────────────────

def process_zip(zip_info, out_dir: str, use_yucca: bool):
    """Process all scans inside one session zip. Returns a status string."""
    pt_name, sub_name, ses_stem, zip_path = zip_info

    try:
        nii_names = _list_niftis_in_zip(zip_path)
    except Exception as e:
        return f"ERR {pt_name}_{sub_name}_{ses_stem}: cannot open zip: {e}"

    if not nii_names:
        return f"ERR {pt_name}_{sub_name}_{ses_stem}: no .nii.gz in zip"

    results = []
    for nii_name in nii_names:
        # Build a unique flat name from the nii path inside the zip:
        #   ses-01/anat/sub-01_ses-01_T1w.nii.gz  →  sub-01_ses-01_T1w
        scan_stem = os.path.splitext(os.path.splitext(os.path.basename(nii_name))[0])[0]
        out_name = f"{pt_name}_{sub_name}_{ses_stem}_{scan_stem}"
        out_path = os.path.join(out_dir, out_name + ".npy")

        if os.path.isfile(out_path):
            results.append(f"SKIP {out_name}")
            continue

        try:
            img = _load_nifti_from_zip(zip_path, nii_name)
            vol = _yucca_preprocess(img) if use_yucca else _simple_preprocess(img)
            np.save(out_path, vol.astype(np.float16))
            results.append(f"OK {out_name}  shape={vol.shape}")
        except Exception as e:
            results.append(f"ERR {out_name}: {e}")

    return "\n".join(results)


# ── scan collection ───────────────────────────────────────────────────────────

def collect_zips(in_path: str, subjects_per_dataset: int = 0):
    """Walk the FOMO300K directory tree and collect (pt, sub, ses, zip_path).

    Handles variable nesting depths across PT datasets — uses rglob so zips
    nested at any depth (pt/sub/ses.zip or pt/group/sub/ses.zip) are found.
    subjects_per_dataset: if > 0, only take the first N unique subject dirs
                          per PT folder (useful for smoke tests).
    """
    items = []
    in_path = Path(in_path)
    for pt_dir in sorted(in_path.iterdir()):
        if not pt_dir.is_dir() or pt_dir.name.startswith("."):
            continue
        all_zips = sorted(pt_dir.rglob("*.zip"))
        if subjects_per_dataset > 0:
            # Limit by unique immediate-child dirs (subject level)
            seen_subs = []
            filtered = []
            for z in all_zips:
                sub = z.relative_to(pt_dir).parts[0]
                if sub not in seen_subs:
                    seen_subs.append(sub)
                if seen_subs.index(sub) < subjects_per_dataset:
                    filtered.append(z)
            all_zips = filtered
        for zip_file in all_zips:
            rel = zip_file.relative_to(pt_dir)
            # Use first path component as sub_name, rest as session stem
            sub_name = rel.parts[0]
            ses_stem = "_".join(rel.parts[1:]).replace(".zip", "")
            items.append((pt_dir.name, sub_name, ses_stem, str(zip_file)))
    return items


# ── parallel runner ───────────────────────────────────────────────────────────

def run(in_path: str, out_path: str, num_workers: int, use_yucca: bool,
        subjects_per_dataset: int = 0):
    maybe_mkdir_p(out_path)

    zips = collect_zips(in_path, subjects_per_dataset=subjects_per_dataset)
    label = f"(first {subjects_per_dataset} subjects/dataset)" if subjects_per_dataset else ""
    print(f"Found {len(zips):,} zip files in {in_path} {label}")

    worker = partial(process_zip, out_dir=out_path, use_yucca=use_yucca)

    from tqdm import tqdm

    if num_workers == 1:
        errors = []
        for item in tqdm(zips, desc="Preprocessing", unit="scan"):
            msg = worker(item)
            if "ERR" in msg:
                errors.append(msg)
                tqdm.write(msg)
        if errors:
            print(f"\n{len(errors)} errors — see above.")
    else:
        from multiprocessing import Pool
        errors = []
        with Pool(num_workers) as pool:
            for msg in tqdm(pool.imap_unordered(worker, zips),
                            total=len(zips), desc="Preprocessing", unit="scan"):
                if "ERR" in msg:
                    errors.append(msg)
                    tqdm.write(msg)
        if errors:
            print(f"\n{len(errors)} errors — see above.")

    # Summary
    done = len(list(Path(out_path).glob("*.npy")))
    print(f"\nDone. {done:,} / {len(zips):,} files in {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_path",     required=True, help="Path to fomo/ directory")
    p.add_argument("--out_path",    required=True, help="Where to write .npy files")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--no_yucca",    action="store_true",
                   help="Use simple preprocessing instead of yucca (faster, no 1mm resampling).")
    p.add_argument("--subjects_per_dataset", type=int, default=0,
                   help="Only process first N subjects per PT dataset. "
                        "0 = all (default). Use 2-3 for local smoke tests.")
    args = p.parse_args()

    use_yucca = HAS_YUCCA and not args.no_yucca
    if not use_yucca:
        print("Warning: using simple preprocessing (no 1mm resampling).")
        if HAS_YUCCA and args.no_yucca:
            print("  (--no_yucca flag set)")
        elif not HAS_YUCCA:
            print("  (yucca not installed; pip install yucca)")

    run(args.in_path, args.out_path, args.num_workers, use_yucca, args.subjects_per_dataset)
