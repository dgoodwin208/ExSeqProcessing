#!/usr/bin/env python

import sys
import re
from datetime import datetime,timedelta
import argparse

parser = argparse.ArgumentParser(description='summary nfsiostat log.')
parser.add_argument('log_file')
parser.add_argument('-s', '--start_time', help='start of effective time')
parser.add_argument('-e', '--end_time', help='end of effective time')
args = parser.parse_args()

logfile = args.log_file

is_all_period = True
start_time = datetime.min
end_time = datetime.max
if args.start_time != None:
    try:
        start_time = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S')
        is_all_period = False
    except ValueError:
        print('start time format is wrong. Format: %Y-%m-%d %H:%M:%S')
        sys.exit(1)
elif args.end_time != None:
    try:
        end_time = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M:%S')
        is_all_period = False
    except ValueError:
        print('end time format is wrong. Format: %Y-%m-%d %H:%M:%S')
        sys.exit(1)


# =============================================================================
comp_fs_path = re.compile('.* mounted on (.*):$')
comp_call_pages = re.compile('^([0-9]*) .* ([0-9]*) pages$')
comp_calls = re.compile('^([0-9]*) .*')
comp_pages_per_call = re.compile('^\(([0-9.]*) pages .*')

# =============================================================================
class StatClass:
    def __init__(self):
        self.clear()

    def clear(self):
        self.op_per_sec = ''
        self.rpc_bklog = ''
        self.read_ops_per_sec = ''
        self.read_kb_per_sec = ''
        self.read_kb_per_op = ''
        self.read_retrans = ''
        self.read_retrans_pc = ''
        self.read_avg_rtt = ''
        self.read_avg_exe = ''
        self.write_ops_per_sec = ''
        self.write_kb_per_sec = ''
        self.write_kb_per_op = ''
        self.write_retrans = ''
        self.write_retrans_pc = ''
        self.write_avg_rtt = ''
        self.write_avg_exe = ''
        self.num_calls_in_readpage = ''
        self.num_pages_in_readpage = ''
        self.num_calls_in_readpages = ''
        self.num_pages_in_readpages = ''
        self.readpages_per_call = ''
        self.num_calls_in_writepage = ''
        self.num_pages_in_writepage = ''
        self.num_calls_in_writepages = ''
        self.num_pages_in_writepages = ''
        self.writepages_per_call = ''

    def print(self, fs_path, base_time, elapsed_time):
        total_hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
        total_minutes, total_seconds = divmod(remainder, 60)
        sys.stdout.write('%s,%d:%02d:%02d,%-8s,' % (base_time + elapsed_time, total_hours, total_minutes, total_seconds, fs_path))
        sys.stdout.write('%8s,%5s,' % (self.op_per_sec, self.rpc_bklog))
        sys.stdout.write('%11s,%11s,%8s,%s,%s,' % (self.read_ops_per_sec, self.read_kb_per_sec, self.read_kb_per_op, self.read_retrans, self.read_retrans_pc))
        sys.stdout.write('%8s,%8s,' % (self.read_avg_rtt, self.read_avg_exe))
        sys.stdout.write('%11s,%11s,%8s,%s,%s,' % (self.write_ops_per_sec, self.write_kb_per_sec, self.write_kb_per_op, self.write_retrans, self.write_retrans_pc))
        sys.stdout.write('%8s,%8s,' % (self.write_avg_rtt, self.write_avg_exe))
        sys.stdout.write('%8s,%8s,' % (self.num_calls_in_readpage, self.num_pages_in_readpage))
        sys.stdout.write('%8s,%10s,%8s,' % (self.num_calls_in_readpages, self.num_pages_in_readpages, self.readpages_per_call))
        sys.stdout.write('%8s,%8s,' % (self.num_calls_in_writepage, self.num_pages_in_writepage))
        sys.stdout.write('%8s,%10s,%8s\n' % (self.num_calls_in_writepages, self.num_pages_in_writepages, self.writepages_per_call))

# =============================================================================

INTERVALS = 5 # sec.
prv_term = ''

stat = StatClass()

sys.stdout.write('datetime,time,fs-path,op/s,rpc-bklog,')
sys.stdout.write('read-ops/s,read-kb/s,read-kb/op,read-retrans,read-retrans(%),read-avg-rtt,read-avg-exe,')
sys.stdout.write('write-ops/s,write-kb/s,write-kb/op,write-retrans,write-retrans(%),write-avg-rtt,write-avg-exe,')
sys.stdout.write('calls-in-readpage,pages-in-readpage,calls-in-readpages,pages-in-readpages,readpages/call,')
sys.stdout.write('calls-in-writepage,pages-in-writepage,calls-in-writepages,pages-in-writepages,writepages/call\n')

fs_path = 'none'
count_table = {}
first_base_time = None
for line in open(logfile, 'r'):
    line = line.rstrip()

    if re.search('^datetime: ', line):
        if fs_path in count_table:
            elapsed_time = count_table[fs_path] * timedelta(seconds=INTERVALS)
            stat.print(fs_path, base_time, elapsed_time)
        fs_path = 'none'
        count_table = {}

        base_time = datetime.strptime(line, 'datetime: %Y/%m/%d %H:%M:%S')
        if first_base_time == None:
            first_base_time = base_time

    elif re.search(' mounted on ', line):
        if fs_path not in count_table:
            count_table[fs_path] = 1
            stat.clear()
        else:
            elapsed_time = count_table[fs_path] * timedelta(seconds=INTERVALS) + (base_time - first_base_time)

            if is_all_period == True or (base_time + elapsed_time >= start_time and base_time + elapsed_time <= end_time):
                stat.print(fs_path, base_time, elapsed_time)

            elif is_all_period == False and base_time + elapsed_time > end_time:
                break

            count_table[fs_path] = count_table[fs_path] + 1

        m = comp_fs_path.search(line)
        if m == None:
            print('mounted fs path regex is wrong.')
            sys.exit(1)

        fs_path = m.group(1)

    elif re.search('rpc bklog', line):
        prv_term = 'rpc bklog'
    elif re.search('^read:', line):
        prv_term = 'read'
    elif re.search('^write:', line):
        prv_term = 'write'
    elif re.search(' nfs_readpage\(\) ', line):
        m = comp_call_pages.search(line)
        if m == None:
            print('call and pages regex is wrong.')
            sys.exit(1)

        stat.num_calls_in_readpage = m.group(1)
        stat.num_pages_in_readpage = m.group(2)

    elif re.search(' nfs_readpages\(\) ', line):
        m = comp_call_pages.search(line)
        if m == None:
            print('call and pages regex is wrong.')
            sys.exit(1)

        stat.num_calls_in_readpages = m.group(1)
        stat.num_pages_in_readpages = m.group(2)

        prv_term = 'readpages'

    elif re.search(' nfs_writepage\(\) ', line):
        m = comp_call_pages.search(line)
        if m == None:
            print('call and pages regex is wrong.')
            sys.exit(1)

        stat.num_calls_in_writepage = m.group(1)
        stat.num_pages_in_writepage = m.group(2)

    elif re.search(' nfs_writepages\(\) ', line):
        m = comp_call_pages.search(line)
        if m == None:
            print('call and pages regex is wrong.')
            sys.exit(1)

        stat.num_calls_in_writepages = m.group(1)
        stat.num_pages_in_writepages = m.group(2)

        prv_term = 'writepages'

    elif re.search(' nfs_updatepage() ', line):
        m = comp_calls.search(line)
        if m == None:
            print('call and pages regex is wrong.')
            sys.exit(1)

        stat.num_calls_in_updatepage = m.group(1)

    else:
        if prv_term == 'rpc bklog':
            values = line.split()
            if len(values) != 2:
                print('# of values is wrong in rpc bklog.')
                sys.exit(1)

            stat.op_per_sec = values[0]
            stat.rpc_bklog = values[1]

            prv_term = ''

        elif prv_term == 'read':
            values = line.split()
            if len(values) != 7:
                print('# of values is wrong in read.')
                sys.exit(1)

            stat.read_ops_per_sec = values[0]
            stat.read_kb_per_sec = values[1]
            stat.read_kb_per_op = values[2]
            stat.read_retrans = values[3]
            stat.read_retrans_pc = values[4][1:-2]
            stat.read_avg_rtt = values[5]
            stat.read_avg_exe = values[6]

            prv_term = ''

        elif prv_term == 'write':
            values = line.split()
            if len(values) != 7:
                print('# of values is wrong in write.')
                sys.exit(1)

            stat.write_ops_per_sec = values[0]
            stat.write_kb_per_sec = values[1]
            stat.write_kb_per_op = values[2]
            stat.write_retrans = values[3]
            stat.write_retrans_pc = values[4][1:-2]
            stat.write_avg_rtt = values[5]
            stat.write_avg_exe = values[6]

            prv_term = ''

        elif prv_term == 'readpages':
            if line == '':
                prv_term = ''
                continue

            m = comp_pages_per_call.search(line)
            if m == None:
                print('readpages regex is wrong.')
                sys.exit(1)

            stat.readpages_per_call = m.group(1)

            prv_term = ''

        elif prv_term == 'writepages':
            if line == '':
                prv_term = ''
                continue

            m = comp_pages_per_call.search(line)
            if m == None:
                print('writepages regex is wrong.')
                sys.exit(1)

            stat.writepages_per_call = m.group(1)

            prv_term = ''


