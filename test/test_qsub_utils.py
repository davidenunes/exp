import unittest


class MyTestCase(unittest.TestCase):
    def test_sgeutils_qsub_params(self):
        qsub_file_conda("test_sub.sh", "/home/davide/dev/test/test.py", "tf", script_params={"f": "shit.txt"})

        # call("qsub test.sh", shell=True)

        """
        the basic idea is to a have an experiment building file that creates
        the parameter space, divides the parameter space into config files
        creates submission files running a given runnable module (param) on the config files
        and possibly some additional parameters
        
        ALTERNATIVE
        
        the experiment building file:
        1. creates a new subfolder on the results folder 
        2. takes the parameter grid, generates the summary file on the result folder
        3. creates one submission file for each possible parameter configuration and sets the results folder 
        as the working directory
        4. it creates a log subfolder and sets the std_out and std_err to deposit the logs in that folder, the path 
        is expanded from the working dir
        
        
        don't forget that log sub-folder must be created
        
        
        import sys; print('Python %s on %s' % (sys.version, sys.platform))
        sys.path.extend(['/home/davex32/MEGA/dev/exp'])
        os.environ["PATH"]
        """

        # path = os.path.abspath(os.path.join(__file__, os.pardir))
        # print("Current Path: ", path)
        # call("cd {path}".format(path=path+"/../.."), shell=True)
        # call("python -m test_script", shell=True)

        # call("python {path}".format(path=path))
        # os.remove("test.sh")


if __name__ == '__main__':
    unittest.main()
