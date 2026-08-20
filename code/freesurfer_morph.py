"""
freesurfer_morph.py — FreeSurfer stats files into a tidy morphometry table

Reads `?h.aparc.stats` and `aseg.stats` from the recon-all delivery and writes
two CSVs into data/:

  freesurfer_morph.csv          one row per subject: DLPFC parcels, global
                                measures, and recon quality flags
  freesurfer_parcels_long.csv   every parcel, both hemispheres, long format

The long file exists so that later questions -- is the E-field deficit
DLPFC-specific or is the whole brain thinner? -- need no re-parsing.

**The DLPFC region matches the E-field ROI exactly.** FreeSurfer's `aparc` is
the Desikan-Killiany atlas, which is the same parcellation SimNIBS ships as
DK40, down to the parcel names. So `rostralmiddlefrontal + caudalmiddlefrontal`
selects literally the same cortex in both pipelines, and structure can be
regressed on field without an ROI-correspondence caveat.

**Thickness is combined across parcels weighted by vertex count**, not as a
plain mean of the two ThickAvg values. Rostral middle frontal is roughly twice
the size of caudal, so an unweighted mean would silently over-weight the
smaller parcel. Vertex weighting also matches how FreeSurfer computes its own
whole-hemisphere MeanThickness.

Usage
-----
    python freesurfer_morph.py
    python freesurfer_morph.py --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import (FREESURFER_ROOT, FREESURFER_MORPH_PATH,
                    FREESURFER_PARCELS_PATH)

# Desikan-Killiany parcels making up the stimulated region. These are the same
# two parcels averaged on the SimNIBS DK40 surface for the parcel-based E-field
# measure, so the two are directly comparable.
DLPFC_PARCELS = ['rostralmiddlefrontal', 'caudalmiddlefrontal']

# Columns of ?h.aparc.stats, per its own ColHeaders line.
APARC_COLS = ['parcel', 'n_vert', 'surf_area', 'gray_vol', 'thick_avg',
              'thick_std', 'mean_curv', 'gaus_curv', 'fold_ind', 'curv_ind']

# aseg.stats "# Measure" keys worth carrying forward, mapped to output names.
# Keyed on the *short* name -- the second comma-separated field -- which is
# what parse_stats_header returns. For most measures the short and long names
# differ only slightly, but eTIV's long name is EstimatedTotalIntraCranialVol,
# so keying on that silently yields NaN.
ASEG_MEASURES = {
    'eTIV': 'etiv',
    'TotalGrayVol': 'total_gray_vol',
    'CortexVol': 'cortex_vol',
    'lhCortexVol': 'lh_cortex_vol',
    'rhCortexVol': 'rh_cortex_vol',
    'SubCortGrayVol': 'subcort_gray_vol',
    'BrainSegVolNotVent': 'brainseg_not_vent',
    'CerebralWhiteMatterVol': 'white_matter_vol',
    'SurfaceHoles': 'surface_holes',
    # Ventricular and total-CSF volume: the standard global markers of the
    # atrophy account, and the ones the original hypothesis rested on. Absent
    # until now, which is why that account was only ever tested via cortical
    # thickness.
    'VentricleChoroidVol': 'ventricle_choroid_vol',
    'SupraTentorialVol': 'supratentorial_vol',
    'SupraTentorialVolNotVent': 'supratentorial_notvent_vol',
    'BrainSegVol': 'brainseg_vol',
    'MaskVol': 'mask_vol',
}


def parse_aparc_stats(path: Path) -> pd.DataFrame:
    """Per-parcel rows from one hemisphere's aparc.stats."""
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) != len(APARC_COLS):
            continue
        rows.append([parts[0]] + [float(v) for v in parts[1:]])
    df = pd.DataFrame(rows, columns=APARC_COLS)
    df['n_vert'] = df['n_vert'].astype(int)
    return df


def parse_stats_header(path: Path) -> Dict[str, float]:
    """
    The `# Measure <key>, <name>, <description>, <value>, <units>` lines.

    Present in both aparc.stats (hemisphere totals) and aseg.stats (global
    volumes). The second field is the short name used here.
    """
    out = {}
    for line in path.read_text().splitlines():
        if not line.startswith('# Measure'):
            continue
        parts = [p.strip() for p in line[len('# Measure'):].split(',')]
        if len(parts) >= 4:
            try:
                out[parts[1]] = float(parts[3])
            except ValueError:
                pass
    return out


def subject_dirs(root: Path = FREESURFER_ROOT) -> List[Path]:
    """
    Every recon-all subject under `root`, across all delivery batches.

    Output arrived in more than one batch, so this walks one level of delivery
    folders as well as the root itself. Anchoring on a single batch folder
    would mean a later delivery is silently skipped and the analysis quietly
    keeps running on the old, smaller sample.
    """
    seen, out = set(), []
    for d in sorted(root.glob('*/sub-*')) + sorted(root.glob('sub-*')):
        if not (d / 'stats').is_dir():
            continue
        sid = subject_id_from_dir(d)
        if sid in seen:
            print(f'  WARNING: sub-{sid} appears in more than one delivery; '
                  f'ignoring the copy in {d.parent.name}')
            continue
        seen.add(sid)
        out.append(d)
    return out


def freesurfer_version(d: Path) -> Optional[str]:
    """The recon-all build stamp, e.g. `...-7.3.2-20220804-6354275`."""
    stamp = d / 'scripts' / 'build-stamp.txt'
    return stamp.read_text().strip() if stamp.exists() else None


def subject_id_from_dir(d: Path) -> str:
    """`sub-11075_ses-01` -> `11075`."""
    return d.name.split('_')[0].replace('sub-', '')


def build_subject_row(d: Path) -> Optional[Dict]:
    """One subject's morphometry, or None if the recon is incomplete."""
    sid = subject_id_from_dir(d)
    aseg = d / 'stats' / 'aseg.stats'
    if not aseg.exists():
        return None

    row: Dict[str, object] = {
        'subject_id': sid,
        'fs_delivery': d.parent.name,
        'fs_version': freesurfer_version(d),
    }

    aseg_hdr = parse_stats_header(aseg)
    for key, name in ASEG_MEASURES.items():
        row[name] = aseg_hdr.get(key)

    for hemi in ('lh', 'rh'):
        stats_path = d / 'stats' / f'{hemi}.aparc.stats'
        if not stats_path.exists():
            return None

        parcels = parse_aparc_stats(stats_path)
        hdr = parse_stats_header(stats_path)
        row[f'{hemi}_mean_thickness'] = hdr.get('MeanThickness')
        row[f'{hemi}_white_surf_area'] = hdr.get('WhiteSurfArea')
        row[f'{hemi}_n_parcels'] = len(parcels)

        sel = parcels[parcels['parcel'].isin(DLPFC_PARCELS)]
        if len(sel) != len(DLPFC_PARCELS):
            return None

        # Vertex-weighted, so the larger parcel counts proportionally. A plain
        # mean of the two ThickAvg values would over-weight caudal, which is
        # roughly half the size of rostral.
        row[f'{hemi}_dlpfc_thickness'] = (
            (sel['thick_avg'] * sel['n_vert']).sum() / sel['n_vert'].sum())
        row[f'{hemi}_dlpfc_gray_vol'] = sel['gray_vol'].sum()
        row[f'{hemi}_dlpfc_surf_area'] = sel['surf_area'].sum()
        row[f'{hemi}_dlpfc_n_vert'] = int(sel['n_vert'].sum())

    return row


def build_parcels_long(dirs: List[Path]) -> pd.DataFrame:
    frames = []
    for d in dirs:
        for hemi in ('lh', 'rh'):
            p = d / 'stats' / f'{hemi}.aparc.stats'
            if not p.exists():
                continue
            f = parse_aparc_stats(p)
            f.insert(0, 'hemi', hemi)
            f.insert(0, 'subject_id', subject_id_from_dir(d))
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run(verbose: bool = True) -> pd.DataFrame:
    dirs = subject_dirs()
    if not dirs:
        raise SystemExit(f'No recon-all output under {FREESURFER_ROOT}')

    rows, skipped = [], []
    for d in dirs:
        r = build_subject_row(d)
        (rows if r else skipped).append(r if r else d.name)

    df = pd.DataFrame(rows)
    # The left hemisphere is the stimulated one; provide it unprefixed so
    # downstream code does not have to remember which side the montage targets.
    for col in ['dlpfc_thickness', 'dlpfc_gray_vol', 'dlpfc_surf_area']:
        df[col] = df[f'lh_{col}']

    df = df.sort_values('subject_id').reset_index(drop=True)
    df.to_csv(FREESURFER_MORPH_PATH, index=False)

    long = build_parcels_long(dirs)
    long.to_csv(FREESURFER_PARCELS_PATH, index=False)

    # Morphometric values are not comparable across FreeSurfer major versions,
    # so a delivery processed with a different build would introduce a batch
    # effect perfectly confounded with whichever subjects it contains. That is
    # invisible once the columns are merged, so it stops the run here.
    versions = df['fs_version'].dropna().unique()
    if len(versions) > 1:
        by_ver = df.groupby('fs_version')['subject_id'].count().to_dict()
        raise SystemExit(
            'Mixed FreeSurfer versions across deliveries -- morphometry is not '
            f'comparable between them:\n  {by_ver}\n'
            'Re-run the odd batch to match, or analyse them separately.')

    if verbose:
        deliveries = df['fs_delivery'].value_counts().to_dict()
        print(f'Parsed {len(df)} subjects from {len(deliveries)} delivery(ies)')
        for name, n in deliveries.items():
            print(f'    {name}: {n}')
        print(f'  FreeSurfer: {versions[0].split("-")[-3] if versions.size else "unknown"}')
        if skipped:
            print(f'  SKIPPED (incomplete): {skipped}')
        print(f'  wrote {FREESURFER_MORPH_PATH.name}  ({df.shape[1]} columns)')
        print(f'  wrote {FREESURFER_PARCELS_PATH.name}  ({len(long)} rows)')

        # Quality flags. Surface holes are topological defects fixed during
        # recon; a high count means the fixer had a lot to do and the surfaces
        # deserve a look before the subject is trusted.
        print('\nQuality checks')
        bad_parcels = df[(df.lh_n_parcels < 34) | (df.rh_n_parcels < 34)]
        print(f'  parcels per hemisphere: {df.lh_n_parcels.min()}-'
              f'{df.lh_n_parcels.max()}'
              f'{"  <- INCOMPLETE" if len(bad_parcels) else ""}')
        if 'surface_holes' in df:
            hi = df.nlargest(3, 'surface_holes')[['subject_id', 'surface_holes']]
            print(f'  surface holes: median {df.surface_holes.median():.0f}, '
                  f'max {df.surface_holes.max():.0f} '
                  f'(sub-{hi.iloc[0].subject_id})')
        print(f'  eTIV range: {df.etiv.min() / 1e3:.0f}k-{df.etiv.max() / 1e3:.0f}k mm3')
        print(f'  lh DLPFC thickness: {df.dlpfc_thickness.min():.2f}-'
              f'{df.dlpfc_thickness.max():.2f} mm '
              f'(mean {df.dlpfc_thickness.mean():.2f})')
    return df


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args(argv)
    run(verbose=not args.quiet)
    return 0


if __name__ == '__main__':
    sys.exit(main())
