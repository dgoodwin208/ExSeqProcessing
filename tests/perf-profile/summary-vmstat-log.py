#!/usr/bin/env python

import sys
import re
from datetime import datetime,timedelta
import argparse

parser = argparse.ArgumentParser(description='summary vmstat log.')
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
if args.end_time != None:
    try:
        end_time = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M:%S')
        is_all_period = False
    except ValueError:
        print('end time format is wrong. Format: %Y-%m-%d %H:%M:%S')
        sys.exit(1)


# =============================================================================


sys.stdout.write('datetime,time,procs-r,procs-b,mem-swpd,mem-free,mem-buff,mem-cache,swap-si,swap-so,io-bi,io-bo,sys-in,sys-cs,cpu-us,cpu-sy,cpu-id,cpu-wa,cpu-st\n')

for line in open(logfile, 'r'):
    line = line.rstrip()

    if re.search('^procs -', line) or re.search(' r  b ', line):
        continue

    else:
        values = line.split()
        if len(values) != 19:
            print('value format is wrong.')
            sys.exit(1)

        cur_time = datetime.strptime(values[17] + " " + values[18], '%Y-%m-%d %H:%M:%S')
        if 'base_time' not in locals():
            base_time = cur_time

        if is_all_period == True or (cur_time >= start_time and cur_time <= end_time):
            elapsed_time = cur_time - base_time
            total_hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
            total_minutes, total_seconds = divmod(remainder, 60)
            sys.stdout.write('%s,%d:%02d:%02d,' % (cur_time, total_hours, total_minutes, total_seconds))
            sys.stdout.write('%3s,%3s,' % (values[0], values[1])) # procs
            sys.stdout.write('%10s,%10s,%10s,%10s,' % (values[2], values[3], values[4], values[5])) # memory
            sys.stdout.write('%4s,%4s,' % (values[6], values[7])) # swap
            sys.stdout.write('%5s,%5s,' % (values[8], values[9])) # io
            sys.stdout.write('%6s,%6s,' % (values[10], values[11])) # system
            sys.stdout.write('%4s,%4s,%4s,%4s\n' % (values[12], values[13], values[14], values[15])) # cpu

        elif is_all_period == False and cur_time > end_time:
            break

