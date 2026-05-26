import threading
import time

import psutil
import pynvml

class MemoryMonitor(threading.Thread):
    def __init__(self,pid):
        super().__init__()
        self.pid = pid
        self.process = psutil.Process(pid)
        self.max_ram_usage = 0
        self.max_vram_usage = 0
        self.running = False
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def run(self):
        self.running = True
        while self.running:
            current_ram_usage = self.process.memory_info().rss
            if current_ram_usage > self.max_ram_usage:
                self.max_ram_usage = current_ram_usage
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            current_vram_usage = 0
            for proc in procs:
                if proc.pid == self.pid:
                    current_vram_usage = proc.usedGpuMemory
            if current_vram_usage is not None and current_vram_usage > self.max_vram_usage:
                self.max_vram_usage = current_vram_usage

            time.sleep(0.001)

    def stop(self):
        self.running = False