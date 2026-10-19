import json
from pathlib import Path
import tempfile
import unittest
from materialize import materialize,digest


class MaterializeTests(unittest.TestCase):
    def test_visibility_relocation_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            r=Path(folder);src=r/'source';files={}
            for name in ('development/training/data','private/suite/labels'):
                p=src/name;p.parent.mkdir(parents=True);p.write_text(name)
                files[name]=dict(bytes=p.stat().st_size,sha256=digest(p))
            m=r/'release.json';m.write_text(json.dumps(dict(schema='pathfinder-resources/v1',files=files)))
            args=(m,digest(m),str(src),r/'visible',r/'hidden')
            materialize(*args)
            self.assertFalse((r/'visible/suite').exists())
            (r/'visible/training/data').write_text('corrupt')
            with self.assertRaises(ValueError):materialize(*args,verify_only=True)
            materialize(*args);materialize(*args,verify_only=True)
            (r/'visible/training/private-leftover').write_text('forbidden stale file')
            with self.assertRaisesRegex(ValueError,'Unlisted'):materialize(*args,verify_only=True)
            (r/'visible/training/private-leftover').unlink()
            with self.assertRaises(ValueError):materialize(m,'0'*64,str(src),r/'visible',r/'hidden')

    def test_path_traversal_and_nested_roots_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            r=Path(folder);m=r/'release.json';m.write_text(json.dumps(dict(schema='pathfinder-resources/v1',files={'development/../escape':{}})))
            with self.assertRaises(ValueError):materialize(m,digest(m),str(r),r/'dev',r/'private')
            with self.assertRaises(ValueError):materialize(m,digest(m),str(r),r/'dev',r/'dev/private')

if __name__=='__main__':unittest.main()
