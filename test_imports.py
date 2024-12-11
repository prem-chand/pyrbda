import unittest


class TestModuleImports(unittest.TestCase):
    def test_imports(self):
        try:
            import numpy as np
            from spatial.jcalc import jcalc
            from spatial.plnr import plnr
            from spatial.rotz import rotz
            from spatial.xlt import xlt
            from spatial.pluho import pluho
            from spatial.mcI import mcI
            from spatial.get_gravity import get_gravity
            from spatial.EnerMo import EnerMo
        except ImportError as e:
            self.fail(f"Import failed: {e}")


if __name__ == '__main__':
    unittest.main()
