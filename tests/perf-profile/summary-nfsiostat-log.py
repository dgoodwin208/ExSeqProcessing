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

INTERVALS = 5 # sec.
prv_term = ''

sys.stdout.write('datetime,time,fs-path,op/s,rpc-bklog,')
sys.stdout.write('read-ops/s,read-kb/s,read-kb/op,read-retrans,read-retrans(%),read-avg-rtt,read-avg-exe,')
sys.stdout.write('write-ops/s,write-kb/s,write-kb/op,write-retrans,write-retrans(%),write-avg-rtt,write-avg-exe,')
sys.stdout.write('calls-in-readpage,pages-in-readpage,calls-in-readpages,pages-in-readpages,readpages/call,')
sys.stdout.write('calls-in-writepage,pages-in-writepage,calls-in-writepages,pages-in-writepages,writepages/call\n')

record_start_time = None
is_first = True
for line in open(logfile, 'r'):
    line = line.rstrip()

    if re.search('^datetime: ', line):
        base_time = datetime.strptime(line, 'datetime: %Y/%m/%d %H:%M:%S')
        count_table = {}

    elif re.search(' mounted on ', line):
        if not is_first:
            if fs_path not in count_table:
                count_table[fs_path] = 0

            accm_time = count_table[fs_path] * timedelta(seconds=INTERVALS)

            if is_all_period == True or (base_time + accm_time >= start_time and base_time + accm_time <= end_time):
                elapsed_time = base_time + accm_time - record_start_time
                total_hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                total_minutes, total_seconds = divmod(remainder, 60)
                sys.stdout.write('%s,%d:%02d:%02d,%-8s,' % (base_time + accm_time, total_hours, total_minutes, total_seconds, fs_path))
                sys.stdout.write('%8s,%5s,' % (op_per_sec, rpc_bklog))
                sys.stdout.write('%11s,%11s,%8s,%s,%s,' % (read_ops_per_sec, read_kb_per_sec, read_kb_per_op, read_retrans, read_retrans_pc))
                sys.stdout.write('%8s,%8s,' % (read_avg_rtt, read_avg_exe))
                sys.stdout.write('%11s,%11s,%8s,%s,%s,' % (write_ops_per_sec, write_kb_per_sec, write_kb_per_op, write_retrans, write_retrans_pc))
                sys.stdout.write('%8s,%8s,' % (write_avg_rtt, write_avg_exe))
                sys.stdout.write('%8s,%8s,' % (num_calls_in_readpage, num_pages_in_readpage))
                sys.stdout.write('%8s,%10s,%8s,' % (num_calls_in_readpages, num_pages_in_readpages, readpages_per_call))
                sys.stdout.write('%8s,%8s,' % (num_calls_in_writepage, num_pages_in_writepage))
                sys.stdout.write('%8s,%10s,%8s\n' % (num_calls_in_writepages, num_pages_in_writepages, writepages_per_call))

            elif is_all_period == False and base_time + accm_time > end_time:
                break

            count_table[fs_path] = count_table[fs_path] + 1
        else:
            record_start_time = base_time
            is_first = False

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

        num_calls_in_readpage = m.group(1)
        num_pages_in_readpage = m.group(2)

    elif re.search(' nfs_readpages\(\) ', line):
        m = comp_call_pages.search(line)
        if m == None:
            print('call and pages regex is wrong.')
            sys.exit(1)

        num_calls_in_readpages = m.group(1)
        num_pages_in_readpages = m.group(2)

        prv_term = 'readpages'

    elif re.search(' nfs_writepage\(\) ', line):
        m = comp_call_pages.search(line)
        if m == None:
            print('call and pages regex is wrong.')
            sys.exit(1)

        num_calls_in_writepage = m.group(1)
        num_pages_in_writepage = m.group(2)

    elif re.search(' nfs_writepages\(\) ', line):
        m = comp_call_pages.search(line)
        if m == None:
            print('call and pages regex is wrong.')
            sys.exit(1)

        num_calls_in_writepages = m.group(1)
        num_pages_in_writepages = m.group(2)

        prv_term = 'writepages'

    elif re.search(' nfs_updatepage() ', line):
        m = comp_calls.search(line)
        if m == None:
            print('call and pages regex is wrong.')
            sys.exit(1)

        num_calls_in_updatepage = m.group(1)

    else:
        if prv_term == 'rpc bklog':
            values = line.split()
            if len(values) != 2:
                print('# of values is wrong in rpc bklog.')
                sys.exit(1)

            op_per_sec = values[0]
            rpc_bklog = values[1]

            prv_term = ''

        elif prv_term == 'read':
            values = line.split()
            if len(values) != 7:
                print('# of values is wrong in read.')
                sys.exit(1)

            read_ops_per_sec = values[0]
            read_kb_per_sec = values[1]
            read_kb_per_op = values[2]
            read_retrans = values[3]
            read_retrans_pc = values[4][1:-2]
            read_avg_rtt = values[5]
            read_avg_exe = values[6]

            prv_term = ''

        elif prv_term == 'write':
            values = line.split()
            if len(values) != 7:
                print('# of values is wrong in write.')
                sys.exit(1)

            write_ops_per_sec = values[0]
            write_kb_per_sec = values[1]
            write_kb_per_op = values[2]
            write_retrans = values[3]
            write_retrans_pc = values[4][1:-2]
            write_avg_rtt = values[5]
            write_avg_exe = values[6]

            prv_term = ''

        elif prv_term == 'readpages':
            if line == '':
                prv_term = ''
                continue

            m = comp_pages_per_call.search(line)
            if m == None:
                print('readpages regex is wrong.')
                sys.exit(1)

            readpages_per_call = m.group(1)

            prv_term = ''

        elif prv_term == 'writepages':
            if line == '':
                prv_term = ''
                continue

            m = comp_pages_per_call.search(line)
            if m == None:
                print('writepages regex is wrong.')
                sys.exit(1)

            writepages_per_call = m.group(1)

            prv_term = ''


