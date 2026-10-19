"""Stream the pinned continuous AxonEM PNG archive to a 16 nm image grid.

Grid origin is the outer voxel corner relative to the unpadded release core.
Missing/filler planes are preserved numerically. image_valid_z.npy separately
marks any output plane whose nonzero interpolation support touches a filler.
This is image availability, never an annotation/GT mask or proof of no defects.
"""
import argparse
from functools import lru_cache
import json
import mmap
from pathlib import Path
import time
import zipfile
import h5py
import numpy as np
from PIL import Image
from fetch_axonem import reserve, usage
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import external, sha256, write_json

ARCHIVE_SHA = '93824734ec7c2c09df87f8b01e32a556446c437988618b27b5234df76659eccf'
ARCHIVE_BYTES = 15207108024
RAW_ORIGIN_NM = np.array([-800, -4096, -4096])
NATIVE_NM = np.array([40, 8, 8])
CORE_EXTENT_NM = np.array([30000, 32768, 32768])


def geometry(origin_nm, size_nm):
    origin = np.asarray(origin_nm, dtype=float)
    if (origin.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(size_nm)
        or size_nm <= 0 or size_nm % 16 != 0 or (origin < 0).any()
        or (origin + size_nm > CORE_EXTENT_NM).any()):
        raise ValueError('ROI must lie inside release core; size must be positive multiple of16nm')
    native_start = (origin - RAW_ORIGIN_NM) / NATIVE_NM
    if not np.array_equal(native_start, np.round(native_start)):
        raise ValueError('ROI origin must align to native voxel outer corners')
    shape = (int(size_nm // 16),) * 3
    if np.prod(shape, dtype=np.int64) > 4_000_000_000:
        raise ValueError('Single output image limited to4GB; choose a smaller continuous ROI')
    return native_start.astype(int), shape


def plane_is_filler(plane):
    """Conservative whole-plane classification; ordinary 128-valued pixels survive."""
    return bool(np.all((plane == 128) | (plane == 0)))


def xy_average(plane, start_yx, shape_yx):
    """Exact 2x linear voxel-center sampling without rounding intermediates."""
    y, x = start_yx; h, w = shape_yx
    crop = plane[y:y + 2*h, x:x + 2*w]
    if crop.shape != (2*h, 2*w):
        raise ValueError('Insufficient real XY context')
    return (crop[::2, ::2].astype(np.float32) + crop[1::2, ::2]
            + crop[::2, 1::2] + crop[1::2, 1::2]) * .25


def interpolate_planes(lower, upper, weight, lower_valid, upper_valid):
    """Round once; float32 accumulation may differ from scipy by one gray level."""
    result = np.floor(lower * (1-weight) + upper * weight + .5).astype(np.uint8)
    valid = (lower_valid or weight == 1) and (upper_valid or weight == 0)
    return result, bool(valid)


def verify_training_alignment(archive, root):
    checks = []
    for name, local_z in [('0-3584-0', 20), ('350-1792-1792', 45), ('700-3584-0', 69)]:
        native_start = np.array(list(map(int, name.split('-'))))
        archive_z = int(native_start[0] + local_z)
        with archive.open(f'im_pad/im{archive_z:04d}.png') as stream:
            image = np.asarray(Image.open(stream))
        path = root / 'data' / f'im_{name}_pad.h5'
        with h5py.File(path) as f:
            training = f['main'][local_z, 640:896, 640:896]
        full = image[native_start[1]+640:native_start[1]+896, native_start[2]+640:native_start[2]+896]
        exact = np.array_equal(training, full)
        if not exact:
            raise ValueError(f'Training/archive pixel alignment failed: {name}')
        checks.append(dict(block=name, local_z=local_z, archive_z=archive_z, crop_yx=[640,896,640,896],
                           pixel_exact=True, training_source_sha256=sha256(path)))
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output-name', default='continuous24')
    p.add_argument('--origin-nm', type=int, nargs=3, default=[3000,4384,4384])
    p.add_argument('--size-nm', type=int, default=24000)
    a = p.parse_args()
    root = external(a.root)
    if Path(a.output_name).name != a.output_name or a.output_name in ('.','..'):
        raise ValueError('Output name must be a new plain directory name')
    start, shape = geometry(a.origin_nm, a.size_nm)
    archive_path = root/'data'/'EM30-M-im-pad.zip'
    if archive_path.stat().st_size != ARCHIVE_BYTES or sha256(archive_path) != ARCHIVE_SHA:
        raise ValueError('Pinned raw archive size/SHA256 mismatch')
    reserve(root, int(np.prod(shape))+shape[0]+100_000_000)
    out = root/a.output_name
    out.mkdir(exist_ok=False)
    begun = time.monotonic()
    with zipfile.ZipFile(archive_path) as archive:
        alignment = verify_training_alignment(archive, root)
        records = {}
        @lru_cache(maxsize=3)
        def plane(index):
            name = f'im_pad/im{index:04d}.png'
            with archive.open(name) as stream:
                raw = np.asarray(Image.open(stream))
            if raw.shape != (5120,5120) or raw.dtype != np.uint8:
                raise ValueError(f'Unexpected PNG image geometry {name}')
            filler = plane_is_filler(raw)
            records[index] = dict(archive_z=index, png=name, png_crc32=archive.getinfo(name).CRC,
                                  filler_0_128_only=filler, minimum=int(raw.min()), maximum=int(raw.max()))
            return xy_average(raw, start[1:], shape[1:]), not filler
        raw_out = np.lib.format.open_memmap(out/'raw.npy', mode='w+', dtype=np.uint8, shape=shape)
        valid = np.ones(shape[0], dtype=bool)
        for z in range(shape[0]):
            native_z = start[0] + (z+.5)*16/40 - .5
            lo = int(np.floor(native_z)); weight = native_z-lo
            low, lv = plane(lo)
            high, hv = plane(lo+1) if weight != 0 else (low,lv)
            raw_out[z], valid[z] = interpolate_planes(low, high, weight, lv, hv)
            if z % 100 == 0:
                raw_out.flush()
                if hasattr(raw_out._mmap, 'madvise'):
                    raw_out._mmap.madvise(mmap.MADV_DONTNEED)
                print(json.dumps(dict(output_plane=z,total_planes=shape[0],seconds=time.monotonic()-begun)),flush=True)
        raw_out.flush()
        np.save(out/'image_valid_z.npy', valid, allow_pickle=False)
        del raw_out
    report = dict(schema='pathfinder-continuous-image/v1',dataset_id='AxonEM-M-author-continuous-raw',
        axes='zyx',coordinate_convention='voxel_outer_corner',origin_nm=a.origin_nm,
        voxel_size_nm=[16,16,16],native_voxel_size_nm=NATIVE_NM.tolist(),shape=list(shape),
        archive_raw_origin_nm=RAW_ORIGIN_NM.tolist(),archive_native_start_zyx=start.tolist(),
        source_url='https://huggingface.co/datasets/pytc/EM30/resolve/63519706548316c1f00863cbea0ece40358a759a/EM30-M-im-pad.zip',
        source_archive_sha256=ARCHIVE_SHA,source_archive_bytes=ARCHIVE_BYTES,source='https://axonem.grand-challenge.org/',
        source_discrepancies=['paper mouse XY7nm; release XY8nm used explicitly',
                              'archive790planes; viewer800planes not substituted'],
        interpolation='trilinear voxel-center mapping; XY2x2 float average then Z blend; uint8 rounding once',
        native_resolution_restored=False,training_alignment=alignment,
        interpolation_rounding_note='float32 separable accumulation may differ by1uint8 gray level from scipy at rounding ties; voxel-center coordinates identical',
        files={name:sha256(out/name) for name in ['raw.npy','image_valid_z.npy']},
        image_valid=dict(layout='z vector broadcasts over y,x',scope='only detected full-plane0/128 filler defects',
            zero_means='one or more nonzero interpolation source weights touched filler; not GT/background',
            valid_planes=int(valid.sum()),invalid_output_planes=np.flatnonzero(~valid).tolist(),
            source_filler_planes=[i for i,row in sorted(records.items()) if row['filler_0_128_only']],
            filler_values_preserved=True,all_source_defects_detected=False),
        source_plane_audit=[v for _,v in sorted(records.items())],
        ground_truth_included=False,paper_comparable=False,
        geometric_full20um_sphere_possible=a.size_nm>=20000,
        full_context_image_validity_verified=False,
        disk_bytes=usage(root),seconds=time.monotonic()-begun)
    write_json(out/'manifest.json',report)
    print(json.dumps({k:report[k] for k in ['shape','origin_nm','disk_bytes','seconds']}),flush=True)


if __name__=='__main__':
    main()
