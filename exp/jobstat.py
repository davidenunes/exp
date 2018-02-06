import sqlite3
from enum import Enum
from datetime import datetime
import time
from collections import deque
from tabulate import tabulate


class JobStatus(Enum):
    STARTED = 0
    COMPLETE = 1
    FAILED = 2


def _to_datetime(str_date):
    convert_str = "%Y-%m-%d %H:%M:%S.%f"
    if str_date is not None:
        return datetime.strptime(str_date, convert_str)
    return None


def print_jobs(jobs):
    """ Prints a table of jobs returned using :func:`jobstat.get_jobs`

    Args:
        jobs: the list with the jobs
    """
    jobs = [(job_id, grid_id, name, JobStatus(status).name) for job_id, grid_id, name, status in jobs]
    print(tabulate(jobs, headers=["job id", "grid_id", "job name", "job status"]))


class JobStat:
    def __init__(self, job_id, start, complete, total_n, n, avg_time):
        self.id = job_id
        self.start = _to_datetime(start)
        self.complete = _to_datetime(complete)
        self.total_n = total_n
        self.n = n
        self.avg_time = avg_time

    @staticmethod
    def from_db(db_record_tuple):
        return JobStat(*db_record_tuple)

    def etc(self):
        """Estimated time of completion"""
        return (self.total_n - self.n) * self.avg_time

    def as_dict(self):
        return {
            "job_id": self.id,
            "start": self.start,
            "complete": self.complete,
            "total_n": self.total_n,
            "n": self.n,
            "avg_time": self.avg_time
        }

    def __str__(self):
        return tabulate([self.as_dict()], headers="keys")

    def to_str(self, headers=False):
        return tabulate([self.as_dict()])


def get_job_stat(conn, job_id):
    c = conn.cursor()
    c.execute('''SELECT * from job_stat WHERE id = ?''', str(job_id))
    job_stat = c.fetchone()
    return job_stat


def get_job_stats(conn, job_ids):
    c = conn.cursor()

    c.execute('''SELECT * from job_stat WHERE id IN {values}'''.format(values=str(tuple(job_ids))))
    stats = c.fetchall()
    return stats


def get_jobs(conn, cols=[], status=None):
    c = conn.cursor()
    if status is None:
        c.execute('''SELECT * FROM job''')
    else:
        c.execute('''SELECT * FROM job WHERE status=?''', status)

    return c.fetchall()


def avg_etc(conn):
    jobs = get_jobs(conn, JobStatus.STARTED.value)
    ids = [id for (id, _, _, _) in jobs]
    job_stats = get_job_stats(conn, ids)
    avg_time = []
    for stat in job_stats:
        job = JobStat.from_db(stat)
        avg_time.append(job.etc())

    return sum(avg_time) / len(avg_time)


def complete_job(conn, job_id):
    cursor = conn.cursor()
    cursor.execute('''UPDATE job 
                      SET status = ?
                      WHERE id = ?
                      ''', (JobStatus.COMPLETE.value, job_id))

    cursor.execute('''UPDATE job_stat 
                          SET completion = ?
                          WHERE id = ?
                          ''', (datetime.now(), job_id))
    conn.commit()


def update_job_stat(conn, job_id, avg_time, n_iter):
    cursor = conn.cursor()
    cursor.execute('''UPDATE job_stat 
                      SET n = ?,
                          avg_time = ?
                          
                      WHERE id = ?
                              ''', (n_iter, avg_time, job_id))

    conn.commit()


def start_job(conn, job_name, grid_id, status, total_n):
    try:
        status = JobStatus(status)
    except ValueError as e:
        raise e

    cursor = conn.cursor()

    if grid_id is None:
        grid_id = "NULL"

    cursor.execute('''INSERT OR IGNORE INTO job VALUES(NULL,?,?,?)''', (grid_id, job_name, status.value))
    job_id = cursor.lastrowid
    start_time = datetime.now()
    if total_n is None:
        total_n = "NULL"

    cursor.execute('''INSERT OR IGNORE INTO job_stat 
                        (id, start, completion, total_n, n, avg_time)
                        VALUES(?,?,NULL,?,0,0)''',
                   (job_id, start_time, total_n))

    conn.commit()

    return job_id


def _create_job_status_tables(conn):
    """ creates the tables ``job`` and ``job_stat`` where the job monitor will
    register times and status for each job.

    Note on job_stat:
        all times are intended to be in seconds (not necessarily integer)

        start: is a datetime for when the job starts
        completion: is a datetime for when the job is completed, remains null until then
        total_n: is the total of iterations for the given job (declared when the job starts, null if not available)
        n: is the number of completed iterations so far, at the end this should be equal to total_n
        avg_time: is the average time of each iteration

        average iteration time should be computed based on a running mean with smoothing
        to ensure we can estimate the time for completion

    Args:
        conn: a sqlite connection
    """
    cursor = conn.cursor()
    cursor.execute('PRAGMA foreign_keys = ON;')

    cursor.execute('''
                    CREATE TABLE IF NOT EXISTS job(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        grid_id INTEGER,
                        name TEXT, 
                        status INTEGER,
                        UNIQUE(grid_id)
                        )
                    ''')

    cursor.execute('''
                    CREATE TABLE IF NOT EXISTS job_stat(
                        id INTEGER,
                        start DATETIME,
                        completion DATETIME,
                        total_n INTEGER,
                        n INTEGER,
                        avg_time REAL,
                        FOREIGN KEY(id) REFERENCES job(id) ON DELETE CASCADE,
                        UNIQUE(id)
                    )''')
    conn.commit()


class JobMonitor:
    def __init__(self, grid_id, db_file="exp_jobs.db", job_name="job", total_n=None, time_window_size=10,
                 time_smooth=0.8):
        self.db_file = db_file
        self.job_name = job_name
        self.grid_id = grid_id
        self.job_id = None
        self.total_n = total_n
        self.time_window = deque([], maxlen=time_window_size)
        self.n = 0
        self.avg_time = None
        self.time_smooth = time_smooth
        self.last_time = 0
        self.conn = sqlite3.connect(db_file)
        _create_job_status_tables(self.conn)

    def start(self):
        self.job_id = start_job(self.conn,
                                self.job_name,
                                self.grid_id,
                                JobStatus.STARTED,
                                self.total_n)

        self.last_time = time.time()

    def update(self, n=1):
        delta_t = time.time() - self.last_time
        delta_it = self.n + n - self.n

        if self.avg_time is None:
            self.avg_time = delta_t / delta_it
        else:
            # smooth by const
            self.avg_time = self.time_smooth * delta_t / delta_it + (1. - self.time_smooth) * self.avg_time

        # update n
        self.n += n
        update_job_stat(self.conn, self.job_id, self.avg_time, self.n)
        self.last_time = time.time()

    def complete(self):
        complete_job(self.conn, self.job_id)

    def close(self):
        self.conn.close()
