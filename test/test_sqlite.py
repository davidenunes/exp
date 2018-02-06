import sqlite3
from tabulate import tabulate
from datetime import datetime

statfile = "stat.db"

conn = sqlite3.connect(statfile)

c = conn.cursor()
# unique makes sure we can use insert or ignore
# also might want to make job id a key with autoincrement and use that instead
c.execute('''CREATE TABLE IF NOT EXISTS job_status
             (job_id INTEGER PRIMARY KEY,
              name TEXT, 
              status INTEGER,
              started DATETIME,
              ended DATETIME,
              UNIQUE(job_id)
              )
              ''')

c.execute('''INSERT OR IGNORE INTO job_status VALUES(1,'job_1',0,?,?)''',
          (datetime.now(), datetime.now()))
c.execute('''INSERT OR IGNORE INTO job_status VALUES(1,'job_2',0,NULL,NULL)''')
c.execute('''INSERT OR IGNORE INTO job_status VALUES(2,'job_3',0,NULL,NULL)''')
c.execute('''INSERT OR IGNORE INTO job_status VALUES(3,'job_4',0,?,?)''', (datetime.now(),datetime.now()))

"job_2 violates unique constraint so it does not insert"

conn.commit()

c.execute('''SELECT * FROM job_status''')

jobs = c.fetchall()

JOB_STATUS = {
    0: "DONE",
    1: "RUNNING",
    -1: "FAILED"
}


# map codes to status


def to_datetime(str_date):
    convert_str = "%Y-%m-%d %H:%M:%S.%f"
    if str_date is not None:
        return datetime.strptime(str_date, convert_str)
    return None


jobs = [(i, name, JOB_STATUS[s], to_datetime(start), to_datetime(end)) for i, name, s, start, end in jobs]

print(tabulate(jobs, headers=["job id", "job name", "job status", "start time", "end time"]))

conn.close()
