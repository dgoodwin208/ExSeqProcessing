#!/usr/bin/env python

import sys
import re
from datetime import datetime,timedelta
import argparse

parser = argparse.ArgumentParser(description='summary gpu log.')
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


for line in open(logfile, 'r'):
    line = line.rstrip()

    terms = line.split(', ')

    if terms[0] == 'timestamp':
        sys.stdout.write('%s,time,%s\n' % (terms[0], ','.join(terms[1:])))
        continue

    cur_time = datetime.strptime(terms[0], '%Y/%m/%d %H:%M:%S.%f')

    if 'base_time' not in locals():
        base_time = cur_time

    if is_all_period == True or (cur_time >= start_time and cur_time <= end_time):
        elapsed_time = cur_time - base_time
        total_hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
        total_minutes, total_seconds = divmod(remainder, 60)
        sys.stdout.write('%s,%d:%02d:%6.3f,%s\n' % (cur_time, total_hours, total_minutes, total_seconds, ','.join(terms[1:])))

    if is_all_period == False and cur_time > end_time:
        break

