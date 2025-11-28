import glob
import argparse
import subprocess
import re
import sys
import psutil
import os
from functools import lru_cache
from wsgiref.simple_server import make_server, WSGIRequestHandler
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
from prometheus_client import make_wsgi_app


GPU_UUID_RE = re.compile(
    '(GPU|MIG)-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})')


@lru_cache(maxsize=100)
def get_username(uid):
    """
    Convert a numerical uid to a username
    """
    command = ['/usr/bin/id', '--name', '--user', '{}'.format(uid)]
    return subprocess.check_output(command).strip().decode()


def cgroup_processes(job_dir):
    """
    Find all the PIDs for a cgroup of a job
    """
    procs = []
    res_uid = -1
    for (path, _, _) in os.walk(job_dir):
        with open(os.path.join(path, "cgroup.procs"), 'r') as fprocs:
            for proc in fprocs.readlines():
                pid = int(proc)
                try:
                    ps = psutil.Process(pid)
                    uid = ps.uids().real
                    if uid != 0:
                        res_uid = uid
                        procs.append(pid)
                except psutil.NoSuchProcess:
                    pass
    return res_uid, procs


def split_range(range_str):
    """"
    split a range such as "0-1,3,5,10-13"
    to 0,1,3,5,10,11,12,13
    """
    ranges = []
    for sub in range_str.split(','):
        if '-' in sub:
            subrange = sub.split('-')
            for i in range(int(subrange[0]), int(subrange[1]) + 1):
                ranges.append(i)
        else:
            ranges.append(int(sub))
    return ranges


def get_env(pid):
    """
    Return the environment variables of a process.

    This is used to retrieve the SLURM_JOB_ACCOUNT, but
    doesn't work for regular system users.
    """
    try:
        ps = psutil.Process(pid)
        return ps.environ()
    except psutil.NoSuchProcess:
        raise ValueError("Could not get environment for {}".format(pid))


def fetch_slurm_account(jobid):
    """
    Use scontrol to get the account for a slurm job.
    """
    out = subprocess.run(
        ["scontrol", "show", "job", jobid],
        capture_output=True, text=True, check=True
    ).stdout
    
    for line in out.split():
        if line.startswith("Account="):
            return line.split("=")[1]

    return None
    

def cgroup_gpus(job_dir, cgroups):
    if cgroups == 1:
        task_file = os.path.join(job_dir, "tasks")
    else:
        cgroup_path = os.path.join(job_dir, "gpu_probe")
        # This will create a new cgroup under the root of the job.
        # This is required for v2 since we can only add tasks to leaf cgroups
        try:
            os.mkdir(cgroup_path)
        except FileExistsError:
            pass
        task_file = os.path.join(cgroup_path, "cgroup.procs")
    try:
        res = subprocess.check_output(
            ["get_gpus.sh", task_file]).strip().decode()
    except FileNotFoundError:
        print('get_gpus.sh not found')
        return []
    finally:
        if cgroups == 2:
            # We can remove a cgroup if no tasks are remaining inside
            os.rmdir(cgroup_path)

    gpus = []

    mig = 'MIG' in res
    for line in res.split('\n'):
        m = GPU_UUID_RE.search(line)
        if mig and m and m.group(1) == 'MIG':
            gpus.append((None, m.group()))
        elif not mig and m and m.group(1) == 'GPU':
            gpu = str(line.split()[1].rstrip(':'))
            gpus.append((gpu, m.group()))
    return gpus


class SlurmJobCollector(object):
    """
    Used by a WSGI application to collect and return stats about currently
    running slurm jobs on a node. This is using the stats from the cgroups
    created by Slurm.
    """

    def __init__(self):
        """

        """
        # Will be auto detected by the exporter
        self.UNSUPPORTED_FEATURES = []

    def collect(self):
        """
        Run a collection cycle and update exported stats
        """
        gauge_memory_usage = GaugeMetricFamily(
            'slurm_job_memory_usage', 'Memory used by a job',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_max = GaugeMetricFamily(
            'slurm_job_memory_max', 'Maximum memory used by a job',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_limit = GaugeMetricFamily(
            'slurm_job_memory_limit', 'Memory limit of a job',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_cache = GaugeMetricFamily(
            'slurm_job_memory_cache', 'bytes of page cache memory',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_rss = GaugeMetricFamily(
            'slurm_job_memory_rss',
            'bytes of anonymous and swap cache memory (includes transparent hugepages).',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_rss_huge = GaugeMetricFamily(
            'slurm_job_memory_rss_huge',
            'bytes of anonymous transparent hugepages',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_mapped_file = GaugeMetricFamily(
            'slurm_job_memory_mapped_file',
            'bytes of mapped file (includes tmpfs/shmem)',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_active_file = GaugeMetricFamily(
            'slurm_job_memory_active_file',
            'bytes of file-backed memory on active LRU list',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_inactive_file = GaugeMetricFamily(
            'slurm_job_memory_inactive_file',
            'bytes of file-backed memory on inactive LRU list',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_unevictable = GaugeMetricFamily(
            'slurm_job_memory_unevictable',
            'bytes of memory that cannot be reclaimed (mlocked etc)',
            labels=['user', 'account', 'slurmjobid'])

        counter_core_usage = CounterMetricFamily(
            'slurm_job_core_usage', 'Cpu usage of cores allocated to a job',
            labels=['user', 'account', 'slurmjobid', 'core'])

        gauge_process_count = GaugeMetricFamily(
            'slurm_job_process_count', 'Number of processes in a job',
            labels=['user', 'account', 'slurmjobid'])
        gauge_threads_count = GaugeMetricFamily(
            'slurm_job_threads_count', 'Number of threads in a job',
            labels=['user', 'account', 'slurmjobid', 'state'])

        counter_process_usage = CounterMetricFamily(
            'slurm_job_process_usage', 'Cpu usage of processes within a job',
            labels=['user', 'account', 'slurmjobid', 'exe'])

        if os.path.exists("/sys/fs/cgroup/memory"):
            cgroups = 1  # we are running cgroups v1
        else:
            cgroups = 2  # we are running cgroups v2

        if cgroups == 1:
            jobs_glob = "/sys/fs/cgroup/memory/slurm/uid_*/job_*"
        else:
            jobs_glob = "/sys/fs/cgroup/system.slice/slurmstepd.scope/job_*"
        for job_dir in glob.glob(jobs_glob):
            job = job_dir.split('/')[-1].split('_')[1]
            uid, procs = cgroup_processes(job_dir)
            if len(procs) == 0:
                continue

            # Job is alive, we can get the stats
            user = get_username(uid)

            account = fetch_slurm_account(job)                    

            with open(os.path.join(job_dir, ('memory.usage_in_bytes' if cgroups == 1 else 'memory.current')), 'r') as f_usage:
                gauge_memory_usage.add_metric(
                    [user, account, job], int(f_usage.read()))
            try:
                with open(os.path.join(job_dir, ('memory.max_usage_in_bytes' if cgroups == 1 else 'memory.peak')), 'r') as f_max:
                    gauge_memory_max.add_metric(
                        [user, account, job], int(f_max.read()))
            except FileNotFoundError:
                # 'memory.peak' is only available in kernel 6.8+
                pass

            with open(os.path.join(job_dir, ('memory.limit_in_bytes' if cgroups == 1 else 'memory.max')), 'r') as f_limit:
                val = f_limit.read()
                if cgroups == 2 and val.strip() == 'max':
                    # unlimited memory
                    gauge_memory_limit.add_metric(
                        [user, account, job], sys.maxsize)
                else:
                    gauge_memory_limit.add_metric(
                        [user, account, job], int(f_limit.read()))

            with open(os.path.join(job_dir, 'memory.stat'), 'r') as f_stats:
                stats = dict(line.split() for line in f_stats.readlines())
            if cgroups == 1:
                gauge_memory_cache.add_metric(
                    [user, account, job], int(stats['total_cache']))
                gauge_memory_rss.add_metric(
                    [user, account, job], int(stats['total_rss']))
                gauge_memory_rss_huge.add_metric(
                    [user, account, job], int(stats['total_rss_huge']))
                gauge_memory_mapped_file.add_metric(
                    [user, account, job], int(stats['total_mapped_file']))
                gauge_memory_active_file.add_metric(
                    [user, account, job], int(stats['total_active_file']))
                gauge_memory_inactive_file.add_metric(
                    [user, account, job], int(stats['total_inactive_file']))
                gauge_memory_unevictable.add_metric(
                    [user, account, job], int(stats['total_unevictable']))
            else:
                gauge_memory_cache.add_metric(
                    [user, account, job], int(stats['file']))
                gauge_memory_rss.add_metric(
                    [user, account, job],
                    int(stats['anon']) + int(stats['swapcached']))
                gauge_memory_rss_huge.add_metric(
                    [user, account, job], int(stats['anon_thp']))
                gauge_memory_mapped_file.add_metric(
                    [user, account, job],
                    int(stats['file_mapped']) + int(stats['shmem']))
                gauge_memory_active_file.add_metric(
                    [user, account, job], int(stats['active_file']))
                gauge_memory_inactive_file.add_metric(
                    [user, account, job], int(stats['inactive_file']))
                gauge_memory_unevictable.add_metric(
                    [user, account, job], int(stats['unevictable']))

            # get the allocated cores
            if cgroups == 1:
                cpuset_path = '/sys/fs/cgroup/cpuset/slurm/uid_{}/job_{}/cpuset.effective_cpus'.format(
                    uid, job)
            else:
                cpuset_path = os.path.join(job_dir, 'cpuset.cpus.effective')

            with open(cpuset_path, 'r') as f_cores:
                cores = split_range(f_cores.read())

            if cgroups == 1:
                # There is no equivalent to this in cgroups v2
                with open('/sys/fs/cgroup/cpu,cpuacct/slurm/uid_{}/job_{}/cpuacct.usage_percpu'.format(uid, job), 'r') as f_usage:
                    cpu_usages = f_usage.read().split()
                    for core in cores:
                        counter_core_usage.add_metric([user, account, job, str(core)],
                                                      int(cpu_usages[core]))
            else:
                # We are running cgroups v2, we can use the cpu.stat file, but we won't get the per-core usage
                # We will fake the per-core usage by dividing the total usage by the number of cores so we can still count it
                # eBPF might be used to get the per-core usage in the future
                with open(os.path.join(job_dir, 'cpu.stat'), 'r') as f_usage:
                    core_count = len(cores)
                    cpu_stat = dict(line.split()
                                    for line in f_usage.readlines())
                    # convert to nanoseconds
                    fake_usage_per_core = int(
                        cpu_stat['usage_usec']) * 1000 / core_count
                    for core in cores:
                        counter_core_usage.add_metric([user, account, job, str(core)],
                                                      fake_usage_per_core)

            processes = 0
            tasks_state = {}
            for proc in procs:
                try:
                    p = psutil.Process(proc)
                    cmdline = p.cmdline()
                except psutil.NoSuchProcess:
                    continue
                if len(cmdline) == 0:
                    # sometimes the cmdline is empty, we don't want to count it
                    continue
                if cmdline[0] == '/bin/bash':
                    if len(cmdline) > 1:
                        if '/var/spool' in cmdline[1] and 'slurm_script' in cmdline[1]:
                            # This is the bash script of the job, we don't want to count it
                            continue
                processes += 1

                for t in p.threads():
                    try:
                        pt = psutil.Process(t.id)
                    except psutil.NoSuchProcess:
                        # The thread disappeared between the time we got the list and now
                        continue
                    pt_status = pt.status()
                    if pt_status in tasks_state:
                        tasks_state[pt_status] += 1
                    else:
                        tasks_state[pt_status] = 1

            for status in tasks_state.keys():
                gauge_threads_count.add_metric(
                    [user, account, job, status], tasks_state[status])
            gauge_process_count.add_metric([user, account, job], processes)

            processes_sum = {}
            for proc in procs:
                # get the counter_process_usage data
                try:
                    p = psutil.Process(proc)
                    with p.oneshot():
                        exe = p.exe()
                    if os.path.basename(exe) in ['ssh', 'sshd', 'bash', 'srun']:
                        # We don't want to count them
                        continue
                    else:
                        t = p.cpu_times().user + p.cpu_times().system + \
                            p.cpu_times().children_user + p.cpu_times().children_system
                        if exe in processes_sum:
                            processes_sum[exe] += t
                        else:
                            processes_sum[exe] = t
                except psutil.NoSuchProcess:
                    continue

            # we only count the processes that used more than 60 seconds of CPU
            processes_sum_filtered = processes_sum.copy()
            for exe in processes_sum.keys():
                if processes_sum[exe] < 60:
                    del processes_sum_filtered[exe]

            for exe in processes_sum_filtered.keys():
                counter_process_usage.add_metric(
                    [user, account, job, exe], processes_sum_filtered[exe])

        yield gauge_memory_usage
        yield gauge_memory_max
        yield gauge_memory_limit
        yield gauge_memory_cache
        yield gauge_memory_rss
        yield gauge_memory_rss_huge
        yield gauge_memory_mapped_file
        yield gauge_memory_active_file
        yield gauge_memory_inactive_file
        yield gauge_memory_unevictable
        yield counter_core_usage
        yield gauge_process_count
        yield gauge_threads_count
        yield counter_process_usage


class NoLoggingWSGIRequestHandler(WSGIRequestHandler):
    """
    Class to remove logging of WSGI
    """

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='Promtheus exporter for jobs running with Slurm \
within a cgroup')
    PARSER.add_argument(
        '--port',
        type=int,
        default=9798,
        help='Collector http port, default is 9798')
    ARGS = PARSER.parse_args()

    APP = make_wsgi_app(SlurmJobCollector())
    HTTPD = make_server('', ARGS.port, APP,
                        handler_class=NoLoggingWSGIRequestHandler)
    HTTPD.serve_forever()
