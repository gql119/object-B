"""Finite benchmark-local read-only samples; no scheduler or GPU work."""
import os
from pathlib import Path
import subprocess
import threading
import time


class GPUTrace:
    fields=('utilization.gpu','memory.used','temperature.gpu','clocks.sm','power.draw',
        'clocks_throttle_reasons.sw_thermal_slowdown','clocks_throttle_reasons.hw_thermal_slowdown',
        'clocks_throttle_reasons.sw_power_cap')
    def __init__(self):
        self.rows=[];self.done=threading.Event();self.thread=None
    def start(self):
        if self.thread is not None:raise RuntimeError('Telemetry already started')
        self.thread=threading.Thread(target=self._collect,name='finite-benchmark-readonly-telemetry',daemon=True)
        self.thread.start()
    def _collect(self):
        prior=None
        while not self.done.is_set():
            row={'monotonic':time.perf_counter()}
            try:
                values=subprocess.check_output(['nvidia-smi','--query-gpu='+','.join(self.fields),
                    '--format=csv,noheader,nounits'],text=True,timeout=15).strip().splitlines()
                if len(values)!=1:raise ValueError('Benchmark expects one physical GPU')
                row['gpu']=dict(zip(self.fields,[x.strip() for x in values[0].split(',')]))
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15)
                row['other_GPU_pids']=[int(x.strip()) for x in raw.splitlines() if x.strip() and int(x.strip())!=os.getpid()]
                cpu=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
                if prior is not None:
                    delta=[b-a for a,b in zip(prior,cpu)];total=max(1,sum(delta))
                    row['host_CPU_busy_percent']=100*(total-delta[3]-delta[4])/total
                    row['host_IOwait_percent']=100*delta[4]/total
                prior=cpu
                row['RAM_available_GiB']=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))/2**20
            except Exception as exc:row['sampling_error']=repr(exc)
            self.rows.append(row)
            self.done.wait(2)
    def stop(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=35)
            if self.thread.is_alive():raise RuntimeError('Finite telemetry subprocess did not exit within timeout')
        return self.rows
