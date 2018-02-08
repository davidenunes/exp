import unittest
from params import ParamSpace
import os


class MyTestCase(unittest.TestCase):
    def test_param_grid(self):
        ps = ParamSpace()
        ps.add_value("p1", True)
        ps.add_list("p2", ["A", "B"])
        ps.add_random("p3", 0, 1, 3)
        print("param space size ", ps.grid_size)

        grid = ps.param_grid()

        for params in grid:
            print(params)

        grid = ps.param_grid()
        grid = list(grid)
        self.assertEqual(len(grid), 1 * 2 * 3)
        self.assertEqual(len(grid), ps.grid_size)

    def test_write_recover(self):
        """ There is one issue with writing the param files which is the fact that these do not preserve the
        value types, this is expected, the only issue was that we need to ensure that we can use np.random.uniform
        so regardless of the add_random and add_range arg types, they will be converted to float parameters

        """
        ps = ParamSpace()
        ps.add_value("p1", True)
        ps.add_list("p2", ["A", "B"])
        ps.add_random("p3", 0, 1, 3, persist=False)

        param_filename = "test.conf"
        ps.write(param_filename)
        self.assertTrue(os.path.exists(param_filename))

        ParamSpace.from_file(param_filename)

        os.remove(param_filename)


if __name__ == '__main__':
    unittest.main()
