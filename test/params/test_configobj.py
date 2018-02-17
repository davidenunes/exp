import unittest
import configobj as cfg
import os


class MyTestCase(unittest.TestCase):
    def test_cfg_list(self):
        config_filename = "test.conf"

        c = cfg.ConfigObj(config_filename)
        c["ls"] = [1, 2, 3, 4, 5, 6]
        c.write()

        c2 = cfg.ConfigObj(config_filename)
        ls = c2["ls"]
        self.assertIsInstance(ls, list)
        self.assertIsInstance(ls[0], str)

        os.remove(config_filename)

    def test_cfg_list(self):
        config_filename = "test.conf"

        c = cfg.ConfigObj(config_filename)
        c["dict"] = {"a": 1, "b": 2}
        c.write()

        c2 = cfg.ConfigObj(config_filename)
        d = c2["dict"]
        print(d)

        os.remove(config_filename)

if __name__ == '__main__':
    unittest.main()
