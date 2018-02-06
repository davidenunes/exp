import unittest
from subprocess import call
import os

from exp.grid import qsub


class MyTestCase(unittest.TestCase):
    def test_sgeutils_qsub_params(self):
        job_params = qsub.sge(run_in_cwd=True,
                              job_name="jabbas",
                              std_out_file="logs/pimba_out.txt",
                              std_err_file="logs/pimba_err.txt")
        # for param in job_params:
        #    print(param)
        print(job_params)

        venv = qsub.conda_activate("tf")
        print(venv)

        with open("test.sh", "w") as sh_file:
            sh_file.writelines(["#!/bin/bash\n"])
            sh_file.writelines(job_params)
            sh_file.writelines([venv])
            sh_file.writelines("pip freeze\n")
            sh_file.writelines(["source deactivate\n"])

        call("qsub test.sh", shell=True)

        os.remove("test.sh")


if __name__ == '__main__':
    unittest.main()
