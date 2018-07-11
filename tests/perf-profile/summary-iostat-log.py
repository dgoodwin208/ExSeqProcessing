#!/usr/bin/env python

import sys
import re
from datetime import datetime,timedelta
import argparse

parser = argparse.ArgumentParser(description='summary iostat log.')
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
comp_hostname = re.compile('.* \(([a-zA-Z0-9]*)\).*')

# =============================================================================


sys.stdout.write('datetime,time,device,rrqm/s,wrqm/s,r/s,w/s,rkB/s,wkB/s,avgrq-sz,avgqu-sz,await,r_await,w_await,svctm,%util\n')

count = 0
for line in open(logfile, 'r'):
    line = line.rstrip()

    if line == '' or re.search('^Device', line):
        continue

    if re.search('Linux ', line):
        m = comp_hostname.search(line)
        if m == None:
            print('hostname regex is wrong.')
            sys.exit(1)

        hostname = m.group(1)
        f = open('hostname.txt', 'w')
        f.write(hostname)
        f.close()

    elif re.search('^[0-9/]* [0-9:]* [AP]M', line):
        cur_time = datetime.strptime(line, '%m/%d/%Y %I:%M:%S %p')

        if 'base_time' not in locals():
            base_time = cur_time

        count = count + 1

    elif count > 1 and (is_all_period == True or (cur_time >= start_time and cur_time <= end_time)):
        elapsed_time = cur_time - base_time
        total_hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
        total_minutes, total_seconds = divmod(remainder, 60)
        values = re.sub(r"([0-9a-zA-Z]) ", r"\1, ", line)
        sys.stdout.write('%s,%d:%02d:%02d,' % (cur_time, total_hours, total_minutes, total_seconds))
        sys.stdout.write('%s\n' % values)

    elif is_all_period == False and cur_time > end_time:
        break

