#!/usr/bin/env python

import sys
import re
from datetime import datetime,timedelta
import argparse
from enum import IntEnum
import getpass

parser = argparse.ArgumentParser(description='summary matlab proceeses in top log.')
parser.add_argument('top_log')
args = parser.parse_args()

topfile = args.top_log


# =============================================================================
comp_time_load_averages = re.compile('top - ([0-9]*:[0-9]*:[0-9]*) .*load average: ([^,]*), ([^,]*), ([^,]*)')
comp_mem = re.compile('(.iB) Mem : *([0-9.]*)[ +]total, *([0-9.]*)[ +]free, *([0-9.]*)[ +]used, *([0-9.]*)[ +]buff/cache')
comp_swap = re.compile('.iB Swap: *([0-9.]*)[ +]total, *([0-9.]*)[ +]free, *([0-9.]*)[ +]used. *([0-9.]*)[ +]avail Mem')

PROC_LEFT_SEP = 67
comp_proc_info = re.compile(' *([0-9]*) *([a-z]*) *[0-9]* *[0-9]* *([0-9.]*)([mgt]?) *([0-9.]*)([mgt]?) *([0-9.]*)([mgt]?) ([A-Z]) *([0-9.]*) *([0-9.]*) *([0-9:.]*)')

PROC_LIST = ['main', 'helper', 'worker', 'smpd', 'mpiexec', 'softgl', 'others']

# =============================================================================
class MemUsage:
    def __init__(self, out_filename):
        self.ofile = open(out_filename, 'w')
        self.clear()

    def __del__(self):
        self.ofile.close()

    def clear(self):
        self.mem_total = 0
        self.mem_free = 0
        self.mem_used = 0
        self.buf_cache = 0
        self.swap_total = 0
        self.swap_free = 0
        self.swap_used = 0
        self.avail_mem = 0
        self.mem_unit = ''

    def set_start_time(self, start_time):
        self.start_time = start_time

    def set_mem(self, mem_total, mem_free, mem_used, buf_cache, mem_unit):
        self.mem_total = mem_total
        self.mem_free = mem_free
        self.mem_used = mem_used
        self.buf_cache = buf_cache
        self.mem_unit = mem_unit

    def set_swap(self, swap_total, swap_free, swap_used, avail_mem):
        self.swap_total = swap_total
        self.swap_free = swap_free
        self.swap_used = swap_used
        self.avail_mem = avail_mem

    def write_headers(self):
        self.ofile.write('datetime,elapsed_time,')
        self.ofile.write('mem_total,mem_free,mem_used,swap_total,swap_free,swap_used,buf_cache,avail_mem,mem_unit')
        self.ofile.write('\n')

    def write_values(self, elapsed_time):
        total_hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
        total_minutes, total_seconds = divmod(remainder, 60)
        self.ofile.write('%s,%d:%02d:%02d,' % (self.start_time + elapsed_time, total_hours, total_minutes, total_seconds))
        self.ofile.write('%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%s' \
            % (self.mem_total, self.mem_free, self.mem_used, \
               self.swap_total, self.swap_free, self.swap_used, \
               self.buf_cache, self.avail_mem, self.mem_unit))
        self.ofile.write('\n')

class ProcStatus:
    def __init__(self, out_filename):
        self.ofile = open(out_filename, 'w')
        self.clear()

    def __del__(self):
        self.ofile.close()

    def clear(self):
        self.proc_table = {x:[] for x in PROC_LIST}

    def set_start_time(self, start_time):
        self.start_time = start_time

    def get_start_time(self):
        return self.start_time

    class ProcTrait:
        def __init__(self, pid, virt, res, shr, status, cpu, mem, rtime):
            self.pid = pid
            self.virt = virt
            self.res = res
            self.shr = shr
            self.status = status
            self.cpu = cpu
            self.mem = mem
            self.rtime = rtime

    def set(self, proc, pid, virt, res, shr, status, cpu, mem, rtime):
        p = ProcStatus.ProcTrait(pid, virt, res, shr, status, cpu, mem, rtime)
        self.proc_table[proc].append(p)

    def write_headers(self):
        self.ofile.write('datetime,elapsed_time,proc,pid,virt,res,shr,cpu,mem\n')

    def write_values(self, elapsed_time):
        for proc in PROC_LIST:
            for p in self.proc_table[proc]:
                total_hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                total_minutes, total_seconds = divmod(remainder, 60)
                self.ofile.write('%s,%d:%02d:%02d,' % (self.start_time + elapsed_time, total_hours, total_minutes, total_seconds))
                self.ofile.write('%-10s,%5s,' % (proc, p.pid))
                self.ofile.write('%9.3f,%9.3f,%9.3f,%5.1f,%5.1f\n' \
                    % (p.virt/1000.0, p.res/1000.0, p.shr/1000.0, p.cpu, p.mem))


# =============================================================================

cur_user = getpass.getuser()
start_time = ""
elapsed_time = timedelta(seconds=0)
#count = 0 # for debug

mem_usage_outfile = 'summary-top-mem-usage.csv'
proc_status_outfile = 'summary-top-proc-status.csv'

mem_usage = MemUsage(mem_usage_outfile)
proc_status = ProcStatus(proc_status_outfile)

mem_usage.write_headers()
proc_status.write_headers()

is_first = True
is_matlab_running = False

RunningState = IntEnum('RunningState', 'NONE START_RUNNING RUNNING JUST_FINISHED FINISHED')
running_state = RunningState.NONE

for line in open(topfile, 'r'):
#    print line,
    line = line.rstrip()

    if running_state == RunningState.FINISHED:
        break

    if re.search('^top -', line):
        m = comp_time_load_averages.search(line)
        if m == None:
            print('load_avr regex is wrong.')
            sys.exit(1)

        cur_time = datetime.strptime(m.group(1), '%H:%M:%S')
        load_avr1 = m.group(2)
        load_avr2 = m.group(3)
        load_avr3 = m.group(4)

        if 'prv_time' not in locals():
            prv_time = cur_time

        if not is_first:
            if running_state <= RunningState.JUST_FINISHED:
                diff_time = cur_time - prv_time
                if diff_time < timedelta(days=0):
                    diff_time = diff_time + timedelta(days=1)

                elapsed_time = elapsed_time + diff_time
                mem_usage.write_values(elapsed_time)
                proc_status.write_values(elapsed_time)
                prv_time = cur_time

            if running_state == RunningState.NONE and is_matlab_running == True:
                print('matlab process starting time: %s (%s)' % (proc_status.get_start_time() + elapsed_time, elapsed_time))
                running_state = RunningState.START_RUNNING
            elif running_state == RunningState.START_RUNNING:
                running_state = RunningState.RUNNING
            elif running_state == RunningState.RUNNING and is_matlab_running == False:
                running_state = RunningState.JUST_FINISHED
            elif running_state == RunningState.JUST_FINISHED:
                print('matlab process finished time: %s (%s)' % (proc_status.get_start_time() + elapsed_time, elapsed_time))
                running_state = RunningState.FINISHED
        else:
            print('top-recording  starting time: %s (%s)' % (proc_status.get_start_time() + elapsed_time, elapsed_time))
            is_first = False

        mem_usage.clear()
        proc_status.clear()
        is_matlab_running = False

#        count = count + 1
#        if count == 20:
#            break

    elif re.search('^.iB Mem', line):
        m = comp_mem.search(line)
        if m == None:
            print('mem regex is wrong.')
            sys.exit(1)

        mem_unit = m.group(1)
        mem_total = float(m.group(2))
        mem_free = float(m.group(3))
        mem_used = float(m.group(4))
        buf_cache = float(m.group(5))

        mem_usage.set_mem(mem_total, mem_free, mem_used, buf_cache, mem_unit)
        continue

    elif re.search('^.iB Swap', line):
        m = comp_swap.search(line)
        if m == None:
            print('swap regex is wrong.')
            sys.exit(1)

        swap_total = float(m.group(1))
        swap_free = float(m.group(2))
        swap_used = float(m.group(3))
        avail_mem = float(m.group(4))

        mem_usage.set_swap(swap_total, swap_free, swap_used, avail_mem)
        continue

    elif re.search('MATLAB', line):
        proc_info = line[0:PROC_LEFT_SEP].lstrip()
        proc_name = line[PROC_LEFT_SEP+1:]

        m = comp_proc_info.search(proc_info)
        if m == None:
            print('proc. info regex is wrong.')
            sys.exit(1)

        try:
            pid = m.group(1)
            user = m.group(2)

            if user != cur_user:
                continue

            virt = float(m.group(3))
            if m.group(4) == 'm':
                virt = virt * 10**3
            elif m.group(4) == 'g':
                virt = virt * 10**6
            elif m.group(4) == 't':
                virt = virt * 10**9
            res = float(m.group(5))
            if m.group(6) == 'm':
                res = res * 10**3
            elif m.group(6) == 'g':
                res = res * 10**6
            elif m.group(6) == 't':
                res = res * 10**9
            shr = float(m.group(7))
            if m.group(8) == 'm':
                shr = shr * 10**3
            elif m.group(8) == 'g':
                shr = shr * 10**6
            elif m.group(8) == 't':
                shr = shr * 10**9
            status = m.group(9)
            cpu = float(m.group(10))
            mem = float(m.group(11))
            rtime = m.group(12)
        except:
            sys.stderr.write('%s\n' % proc_info)

        proc = ''
        if re.search('matlab_helper', proc_name):
            proc = 'helper'
        elif re.search('-dmlworker', proc_name):
            proc = 'worker'
        elif re.search('-nosplash', proc_name):
            proc = 'main'
        elif re.search('-prefer', proc_name):
            proc = 'softgl'
        elif re.search('smpd', proc_name):
            proc = 'smpd'
        elif re.search('mpiexec', proc_name):
            proc = 'mpiexec'
        elif re.search('defunct', proc_name):
            continue
        elif re.search('psname.sh', proc_name):
            continue
        else:
            proc = 'others'
            print(line)
#        print(line)
#        print('pid=%s,user=%s,proc=%s' % (pid, user, proc))

        proc_status.set(proc, pid, virt, res, shr, status, cpu, mem, rtime)
        is_matlab_running = True

    elif 'datetime: ' in line:
        count = 0
        start_time = datetime.strptime(line, 'datetime: %Y/%m/%d %H:%M:%S')
        mem_usage.set_start_time(start_time)
        proc_status.set_start_time(start_time)

