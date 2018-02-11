import unittest
from exp import qsub

class MyTestCase(unittest.TestCase):
    def test_pythonpath(self):
        path = qsub.pythonpath_add(__file__)
        print(path)


if __name__ == '__main__':
    unittest.main()
