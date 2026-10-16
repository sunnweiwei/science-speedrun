"""Process-local catalog footprint repair for preserved failed acquisitions.

Only invalid footprints are repaired with Shapely make_valid before the official
rslearn clipping/reprojection. No scene is discarded and valid geometry is
unchanged. Original/repaired WKB and validity reasons remain in a JSONL receipt.
"""

import hashlib
import json
import os
from pathlib import Path

from shapely import make_valid
from shapely.validation import explain_validity
from rslearn.utils import STGeometry
import rslearn.data_sources.utils as matching

from materialize_external_lfmc import main


def repair_geometries(geometries, log_path):
    repaired = []
    for geometry in geometries:
        if geometry.shp.is_valid:
            repaired.append(geometry)
            continue
        shape = make_valid(geometry.shp)
        if not shape.is_valid or shape.is_empty or shape.area <= 0:
            raise ValueError("Catalog footprint repair did not produce a valid area")
        record = {
            "reason": explain_validity(geometry.shp),
            "original_wkb_hex": geometry.shp.wkb_hex,
            "repaired_wkb_hex": shape.wkb_hex,
            "original_sha256": hashlib.sha256(geometry.shp.wkb).hexdigest(),
            "repaired_sha256": hashlib.sha256(shape.wkb).hexdigest(),
            "projection": str(geometry.projection),
            "method": "shapely.make_valid default linework; no scene filtering",
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        repaired.append(STGeometry(geometry.projection, shape, geometry.time_range))
    return repaired


if __name__ == "__main__":
    original = matching.safely_reproject_within_valid_area
    log_path = Path(os.environ["LFMC_GEOMETRY_REPAIR_LOG"])

    def repaired_reproject(geometries, valid_geometry):
        return original(repair_geometries(geometries, log_path), valid_geometry)

    matching.safely_reproject_within_valid_area = repaired_reproject
    main()
