import unittest
from exp.params import ParamSpace, Types, DTypes

import csv
import os


class MyTestCase(unittest.TestCase):
    def test_param_grid(self):
        ps = ParamSpace()
        ps.add_value("p1", True)
        ps.add_list("p2", ["A", "B"])
        ps.add_random("p3", low=0, high=4, prior="uniform", n=3)
        # print("param space size ", ps.grid_size)

        grid = ps.param_grid()

        # for params in grid:
        #    print(params)

        grid = ps.param_grid()
        grid = list(grid)
        self.assertEqual(len(grid), 1 * 2 * 3)
        self.assertEqual(len(grid), ps.size)

    def test_write_recover(self):
        """ There is one issue with writing the param assets which is the fact that these do not preserve the
        value types, this is expected, the only issue was that we need to ensure that we can use np.random.uniform
        so regardless of the add_random and add_range arg types, they will be converted to float parameters
        """
        ps = ParamSpace()
        ps.add_value("p1", True)
        ps.add_list("p2", ["A", "B"])
        ps.add_random("p3", low=0, high=4, prior="uniform", n=3)

        param_filename = "test.conf"
        ps.write(param_filename)
        self.assertTrue(os.path.exists(param_filename))
        ParamSpace(param_filename)
        os.remove(param_filename)

    def test_write_summary(self):
        summary_file = "params.csv"

        ps = ParamSpace()
        ps.add_value("p1", True)
        ps.add_list("p2", ["A", "B"])
        ps.add_random("p3", low=0, high=4, prior="uniform", n=3)
        # print("param space size ", ps.grid_size)

        ps.write_configs(summary_file)

        written_summary = open(summary_file)
        reader = csv.DictReader(written_summary)

        params = [dict(config) for config in reader]
        # print("read parameters")
        # for config in params:
        #    print(config)

        written_summary.close()
        os.remove(summary_file)

        self.assertEqual(len(params), ps.size)

    def test_add_random(self):
        """ If persist is not set to True for add_random
        each time we call param_grid, it samples new random values
        this is because persist = True saves the parameter as a list
        or randomly generated parameters
        """
        ps = ParamSpace()
        name = "param1"
        ps.add_random(name, low=2, high=4, persist=False, n=10, prior="uniform")

        params1 = ps.param_grid()
        self.assertTrue(ps.size, 1)
        r1 = next(params1)[name]

        params2 = ps.param_grid()
        r2 = next(params2)[name]

        ps.write("test.cfg")

        self.assertNotEqual(r1, r2)

    def test_add_range(self):
        filename = "test.cfg"
        ps = ParamSpace()
        ps.add_range("range_param", 0, 10, 1, dtype=int)

        ps.write(filename)

        ps = ParamSpace(filename)
        # print(ps.params["range_param"])
        # print(ps.get_range("range_param"))

        os.remove(filename)

    def test_domain(self):
        ps = ParamSpace()
        ps.add_value("value", True)
        domain = ps.domain("value")
        self.assertIn("domain", domain)
        self.assertIn("dtype", domain)
        self.assertEqual(DTypes.CATEGORICAL.value, domain["dtype"])

        ps.add_list("bool", [True, False, True])
        domain = ps.domain("bool")
        self.assertIn("domain", domain)
        self.assertIn("dtype", domain)
        self.assertEqual(DTypes.CATEGORICAL.value, domain["dtype"])
        self.assertListEqual([True, False], domain["domain"])

        ps.add_range("bounds", 0, 10, dtype=float)
        domain = ps.domain("bounds")
        self.assertIn("domain", domain)
        self.assertIn("dtype", domain)
        self.assertIn("prior", domain)
        self.assertEqual("float", domain["dtype"])
        self.assertEqual("uniform", domain["prior"])

        ps.add_random("random", 0, 10, prior="log-uniform", dtype=float)
        domain = ps.domain("bounds")
        self.assertIn("domain", domain)
        self.assertIn("dtype", domain)
        self.assertIn("prior", domain)
        self.assertEqual("float", domain["dtype"])
        self.assertEqual("uniform", domain["prior"])

    def test_param_grid_with_id(self):
        ps = ParamSpace()
        ps.add_value("p1", True)
        ps.add_list("p2", ["A", "B"])

        params1 = ps.param_grid(runs=5)

        self.assertEqual(len(list(params1)), 1 * 2 * 5)

    def test_write_grid_files(self):
        ps = ParamSpace()
        ps.add_value("p1", True)
        ps.add_list("p2", ["A", "B"])
        ps.add_random("p3", n=2, prior="uniform", low=1, high=3)
        # print("param space size ", ps.grid_size)

        out_path = "/tmp/test_params/"
        if not os.path.exists(out_path) or not os.path.isdir(out_path):
            os.makedirs(out_path)
        ps.write_config_files(out_path)

    def test_sample_params(self):
        ps = ParamSpace()
        ps.add_value("p1", True)
        ps.add_list("p2", ["A", "B"])
        ps.add_random("p3", n=1, prior="uniform", low=1, high=3)

        x = ps.sample_space()
        self.assertIsInstance(x, dict)


if __name__ == '__main__':
    unittest.main()
