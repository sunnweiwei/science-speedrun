"""Behaviour of the MatterGen ordered-disordered matcher shipped in lemat-substitution-novelty.patch.

Needs pymatgen (the scorer environment); skipped elsewhere. The module under test is read from the
patch itself, so the test exercises the exact bytes the scorer image applies.
"""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

TASK = Path(__file__).resolve().parents[1]
PATCH = TASK / "tests/runtime/lemat-substitution-novelty.patch"

try:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Lattice, Structure
except ImportError:  # pragma: no cover - fast contract environment without pymatgen
    Structure = None


def patched_module():
    lines, inside = [], False
    for line in PATCH.read_text().splitlines():
        if line.startswith("+++ "):
            inside = line == "+++ b/src/lemat_genbench/fingerprinting/substitution.py"
            continue
        if inside:
            if line.startswith("diff --git"):
                break
            if line.startswith("+"):
                lines.append(line[1:])
    directory = tempfile.mkdtemp(prefix="mattergen-matcher-")
    path = Path(directory) / "crystalgen_substitution.py"
    path.write_text("\n".join(lines) + "\n")
    spec = importlib.util.spec_from_file_location("crystalgen_substitution", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fcc_binary(first, second, layers):
    """Eight-site fcc cell (conventional cell doubled along c); `layers` gives the species of each (001) plane."""
    a = 3.9
    sites = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.25), (0, 0.5, 0.25),
             (0, 0, 0.5), (0.5, 0.5, 0.5), (0.5, 0, 0.75), (0, 0.5, 0.75)]
    planes = [0, 0, 1, 1, 2, 2, 3, 3]
    species = [first if layers[plane] == "A" else second for plane in planes]
    return Structure(Lattice.tetragonal(a, 2 * a), species, sites)


def rocksalt_with_dopants(dopant_sites):
    """Rock-salt DyN cell doubled along a (8 Dy, 8 N) with Gd on the given cation sites."""
    a = 4.9
    cations = [(0, 0, 0), (0.25, 0.5, 0), (0.25, 0, 0.5), (0, 0.5, 0.5)]
    cations += [(x + 0.5, y, z) for x, y, z in cations]
    anions = [(x + 0.25, y, z) for x, y, z in cations]
    species = ["Gd" if index in dopant_sites else "Dy" for index in range(8)] + ["N"] * 8
    return Structure(Lattice.orthorhombic(2 * a, a, a), species, cations + anions)


@unittest.skipIf(Structure is None, "pymatgen is not installed in this environment")
class MatterGenMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = patched_module()
        class LegacyMatcher(StructureMatcher):
            is_equivalent = StructureMatcher.fit
        cls.matcher = cls.module.DisorderAwareMatcher(LegacyMatcher(ltol=0.1))
        cls.lemat = StructureMatcher(ltol=0.1)

    def test_patch_keeps_mattergen_thresholds_and_dilute_guard(self):
        text = PATCH.read_text()
        self.assertIn("DILUTE_FRACTION = 0.25", text)
        self.assertIn("dilute_variants", text)
        matcher = self.module.DefaultDisorderedStructureMatcher()
        self.assertEqual((matcher.ltol, matcher.stol, matcher.angle_tol), (0.2, 0.3, 5.0))
        self.assertTrue(matcher._primitive_cell and matcher._scale and matcher._supercell and matcher._subset)
        self.assertEqual((matcher.relative_radius_difference_threshold,
                          matcher.electronegativity_difference_threshold), (0.3, 1.0))
        self.assertEqual((matcher.reduced_formula_atol, matcher.reduced_formula_rtol), (1e-2, 1e-1))
        self.assertEqual(text.count("DisorderAwareMatcher(self.fingerprinter)"), 2)

    def test_two_orderings_of_a_half_half_binary_are_one_structure(self):
        # Cu and Au pass the rule (equal radii, electronegativity difference 0.64).
        alternating = fcc_binary("Cu", "Au", "ABAB")
        paired = fcc_binary("Cu", "Au", "AABB")
        self.assertFalse(self.lemat.fit(alternating, paired), "orderings must differ as ordered structures")
        self.assertFalse(self.module.OrderedStructureMatcher().fit(alternating, paired))
        self.assertTrue(self.matcher.is_equivalent(alternating, paired))
        self.assertTrue(self.matcher.is_equivalent(paired, alternating))

    def test_pair_failing_the_hume_rothery_rule_stays_distinct(self):
        # Li-Au: electronegativity difference 1.56 > 1.0. Na-Cs: radius difference 0.36 > 0.3.
        for first, second in (("Li", "Au"), ("Na", "Cs")):
            element_first, element_second = (Structure(Lattice.cubic(3), [x], [[0, 0, 0]]).species[0]
                                             for x in (first, second))
            self.assertFalse(self.module.do_elements_substitute(element_first, element_second))
            alternating = fcc_binary(first, second, "ABAB")
            paired = fcc_binary(first, second, "AABB")
            self.assertFalse(self.matcher.is_equivalent(alternating, paired))
            self.assertFalse(self.matcher.is_equivalent(paired, alternating))

    def test_xef2_self_matching_and_missing_element_properties(self):
        from types import SimpleNamespace
        from pymatgen.core import Element
        xe, fluorine = Element("Xe"), Element("F")
        self.assertIsNone(xe.atomic_radius)
        crystal = Structure(Lattice.cubic(6.0), ["Xe", "F", "F"],
                            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.2], [0.5, 0.5, 0.8]])
        translated = crystal.copy()
        translated.translate_sites(range(3), [0.15, 0.2, 0.25])
        for first, second in [(crystal, translated), (translated, crystal)]:
            self.assertTrue(self.matcher.is_equivalent(first, second))
        self.assertTrue(self.module.do_elements_substitute(xe, xe))
        for missing in [xe, SimpleNamespace(atomic_radius=1., X=None),
                        SimpleNamespace(atomic_radius=1., X=float("nan"))]:
            self.assertFalse(self.module.do_elements_substitute(missing, fluorine))
            self.assertFalse(self.module.do_elements_substitute(fluorine, missing))
        first, second = fcc_binary("Xe", "F", "ABAB"), fcc_binary("Xe", "F", "AABB")
        self.assertFalse(self.matcher.is_equivalent(first, second))
        self.assertFalse(self.matcher.is_equivalent(second, first))

    def test_dilute_variants_of_one_parent_still_merge(self):
        # Two Gd on Dy sites of DyN (a quarter of the cations), adjacent or far apart.
        near = rocksalt_with_dopants({0, 1})
        far = rocksalt_with_dopants({0, 7})
        self.assertFalse(self.lemat.fit(near, far), "the two dopant arrangements must differ as ordered structures")
        self.assertFalse(self.module.OrderedStructureMatcher().fit(near, far))
        self.assertTrue(self.matcher.is_equivalent(near, far))

    def test_dilute_dopant_of_a_known_parent_is_the_same_structure(self):
        parent = rocksalt_with_dopants(set())
        variant = rocksalt_with_dopants({0})
        parent.replace_species({"N": "Cu"})
        variant.replace_species({"N": "Cu"})
        self.assertFalse(self.module.DefaultDisorderedStructureMatcher().fit(parent, variant))
        self.assertTrue(self.matcher.is_equivalent(variant, parent))
        self.assertTrue(self.matcher.is_equivalent(parent, variant))
        self.assertTrue(any(self.module.DefaultDisorderedStructureMatcher().fit(v, parent)
                            for v in self.module.dilute_variants(variant)))

    def test_memorised_copies_supercells_and_translations_match(self):
        original = rocksalt_with_dopants(set())
        copy = original.copy()
        copy.translate_sites(range(len(copy)), [0.5, 0.25, 0.125])
        copy.make_supercell([1, 2, 1])
        self.assertTrue(self.matcher.is_equivalent(original, copy))

    def test_equal_compositions_match_the_published_method(self):
        full = self.module.DefaultDisorderedStructureMatcher()
        for a,b in (("Cu","Au"),("Li","Au")):
            first = fcc_binary(a,b,"AABB")
            second = fcc_binary(a,b,"ABAB")
            self.assertEqual(self.matcher.is_equivalent(first,second), full.fit(first,second))


if __name__ == "__main__":
    unittest.main()
