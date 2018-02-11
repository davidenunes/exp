import unittest
from exp.params import ParamSpace

import csv
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

    def test_write_summary(self):
        summary_file = "params.csv"

        ps = ParamSpace()
        ps.add_value("p1", True)
        ps.add_list("p2", ["A", "B"])
        ps.add_random("p3", 0, 1, 3)
        print("param space size ", ps.grid_size)

        ps.write_grid_summary(summary_file)

        written_summary = open(summary_file)
        reader = csv.DictReader(written_summary)

        params = [dict(config) for config in reader]
        print("read parameters")
        for config in params:
            print(config)

        written_summary.close()
        os.remove(summary_file)

        self.assertEqual(len(params), ps.grid_size)

    def test_add_random(self):
        """ If persist is not set to True for add_random
        each time we call param_grid, it samples new random values
        this is because persist = True saves the parameter as a list
        or randomly generated parameters
        """
        ps = ParamSpace()
        ps.add_random("rand", 0, 1, 1, persist=False)

        params1 = ps.param_grid()
        self.assertTrue(ps.grid_size, 1)
        r1 = next(params1)["rand"]

        params2 = ps.param_grid()
        r2 = next(params2)["rand"]

        self.assertNotEqual(r1, r2)

    def test_param_grid_with_id(self):
        ps = ParamSpace()
        ps.add_value("p1", True)
        ps.add_list("p2", ["A", "B"])

        params1 = ps.param_grid()
        params1_ls = []
        for i, param in enumerate(params1):
            param.update({"id": i})
            params1_ls.append(param)

        params2 = ps.param_grid(include_id=True)
        self.assertListEqual(params1_ls, list(params2))

    def test_write_grid_files(self):
        ps = ParamSpace()
        ps.add_value("p1", True)
        ps.add_list("p2", ["A", "B"])
        ps.add_random("p3", 0, 1, 3)
        print("param space size ", ps.grid_size)

        out_path = "/tmp/test_params/"
        if not os.path.exists(out_path) or not os.path.isdir(out_path):
            os.makedirs(out_path)
        ps.write_config_files(out_path)


if __name__ == '__main__':
    unittest.main()