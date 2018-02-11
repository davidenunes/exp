import unittest
from exp import qsub


class MyTestCase(unittest.TestCase):
    def test_sgeutils_qsub_params(self):
        qsub.qsub_file_conda("test_qsub.sh",
                             script_path="/home/davide/dev/test/test.py",
                             env_name="tf",
                             script_params={"conf": "params.conf"},
                             pythonpath=["/home/davide/"],
                             sge_params="V")


if __name__ == '__main__':
    unittest.main()
