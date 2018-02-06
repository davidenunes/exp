import unittest
import sqlite3
import time
import os
import random
from exp.jobstat import JobMonitor, get_jobs, print_jobs, JobStat, get_job_stat, avg_etc


class TestJobMonitor(unittest.TestCase):
    def test_init(self):
        monitor_file = "jobs.db"
        conn = sqlite3.connect(monitor_file)
        try:
            monitor = JobMonitor(1, db_file=monitor_file, total_n=10, time_smooth=1.0)
            self.assertTrue(os.path.exists(monitor_file))

            monitor.start()

            jobs = get_jobs(conn)
            print_jobs(jobs)
            monitor.complete()

            jobs = get_jobs(conn)
            print_jobs(jobs)

            job_stat = get_job_stat(conn, 1)
            job_stat = JobStat(*job_stat)
            print(job_stat)
            self.assertEqual(job_stat.total_n, monitor.total_n)
            print(job_stat)
        except Exception as e:
            raise e
        finally:
            os.remove(monitor_file)
            conn.close()

    def test_update(self):
        num_iter = 10
        monitor_file = "jobs.db"
        conn = sqlite3.connect(monitor_file)
        try:
            monitor = JobMonitor(1, db_file=monitor_file, total_n=num_iter, time_smooth=0.3)
            monitor.start()

            start = time.time()
            estimates = []
            for i in range(num_iter):
                time.sleep(1 + random.normalvariate(0, 0.1))
                monitor.update()
                job_stat = JobStat.from_db(get_job_stat(conn, monitor.job_id))
                estimates.append(job_stat.etc())
                print("time estimate = {t}".format(t=job_stat.etc()))

            duration = time.time() - start
            print("Took {t} secs".format(t=duration))
        except Exception as e:
            raise e
        finally:
            os.remove(monitor_file)

    def test_average_etc(self):
        monitor_file = "jobs.db"
        conn = sqlite3.connect(monitor_file)

        try:
            monitor = JobMonitor(1, db_file=monitor_file, total_n=2, time_smooth=0.3)
            monitor.start()
            monitor.update()

            monitor2 = JobMonitor(2, db_file=monitor_file, total_n=2)
            monitor2.start()
            time.sleep(0.5)
            monitor2.update()

            etc = avg_etc(conn)
            self.assertAlmostEqual(etc, 0.25, places=2)
        except:
            raise
        finally:
            os.remove(monitor_file)


if __name__ == '__main__':
    unittest.main()
