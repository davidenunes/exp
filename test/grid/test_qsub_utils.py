import unittest
from exp.grid import qsub


class MyTestCase(unittest.TestCase):
    def test_sgeutils_qsub_params(self):

        qsub.write_qsub_file("test_qsub.sh",
                             script_path="/home/davide/dev/test/test.py",
                             venv="virtualenv",
                             venv_root="esbanhanhas",
                             venv_name="test",
                             script_params={"conf": "params.conf"},
                             pythonpath=["/home/davide/"],
                             sge_params="V",
                             resource_dict={"gpu":"1","release":"el7"})


if __name__ == '__main__':
    unittest.main()
