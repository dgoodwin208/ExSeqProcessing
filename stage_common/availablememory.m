function ret = availablememory()

    [status,cmdout] = system('free -m | sed -ne "s/Mem: .* \([0-9]*\)$/\1/p"');

    ret = str2num(cmdout);
end
