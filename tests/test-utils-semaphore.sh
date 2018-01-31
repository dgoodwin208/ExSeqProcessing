#!/bin/bash
# file: tests/test-utils-semaphore.sh

if [ -f tests/test-helper.sh ]; then
    . tests/test-helper.sh
elif [ -f ./test-helper.sh ]; then
    . ./test-helper.sh
else
    echo "no test-helper.sh in tests dir or current dir."
    exit 1
fi

SHUNIT2_SRC_DIR=/mp/nas1/share/lib/shunit2
SEM=utils/semaphore/semaphore

# =================================================================================================
oneTimeSetUp() {
    if [ -n "$($SEM /t,/t0,/t1 getvalue | grep OK)" ]; then
        $SEM /t,/t0,/t1 unlink > /dev/null 2>&1

        if [ -n "$($SEM /t,/t0,/t1 getvalue | grep OK)" ]; then
            echo "use sudo to remove semaphores initially."
            exit 1
        fi
    fi

    Result_dir=test-results/test-utils-semaphore-$(date +%Y%m%d_%H%M%S)
    mkdir -p $Result_dir

}

oneTimeTearDown() {
    if [ -n "$($SEM /t,/t0,/t1 getvalue | grep OK)" ]; then
        $SEM /t,/t0,/t1 unlink > /dev/null 2>&1
    fi
}

tearDown() {
    $SEM /t,/t0,/t1 unlink > /dev/null 2>&1
}

# =================================================================================================
# normal test cases

test001_open() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    $SEM /t open 3 > $Log 2>&1
    local status=$?
    assertEquals 0 $status
}

test002_close() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    $SEM /t open 3 > $Log 2>&1
    $SEM /t close >> $Log 2>&1
    local status=$?
    assertEquals 0 $status
}

test003_unlink() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    $SEM /t open 3 > $Log 2>&1
    $SEM /t unlink >> $Log 2>&1
    local status=$?
    assertEquals 0 $status
}

test004_getvalue() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    $SEM /t open 3 > $Log 2>&1
    $SEM /t getvalue >> $Log 2>&1
    local status=$?
    assertEquals 0 $status

    local ret=$(cat $Log | sed -ne "s/\[\/t\].*\(OK value=3\)/\1/p")
    assertEquals 'OK value=3' "$ret"
}

test005_wait_and_post() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    $SEM /t open 3 > $Log 2>&1
    $SEM /t wait >> $Log 2>&1
    local status=$?
    assertEquals 0 $status

    $SEM /t getvalue >> $Log 2>&1

    $SEM /t post >> $Log 2>&1
    status=$?
    assertEquals 0 $status

    $SEM /t getvalue >> $Log 2>&1

    local ret=$(cat $Log | sed -ne "s/\[\/t\].*\(OK value=[23]\)/\1/p" | tr -d "\n")
    assertEquals 'OK value=2OK value=3' "$ret"
}

test006_trywait() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    $SEM /t open 3 > $Log 2>&1
    $SEM /t trywait >> $Log 2>&1
    local status=$?
    assertEquals 0 $status

    $SEM /t getvalue >> $Log 2>&1

    local ret=$(cat $Log | sed -ne "s/\[\/t\].*\(OK value=2\)/\1/p")
    assertEquals 'OK value=2' "$ret"
}

test007_exclusive() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    $SEM /t open 2 > $Log 2>&1

    for((i=0;i<3;i++)); do
        {
            sleep $i
            $SEM /t trywait >> $Log 2>&1
            local status=$?

            if [ $i -eq 0 ] || [ $i -eq 1 ]; then
                assertEquals 0 $status

                sleep 4
                $SEM /t post >> $Log 2>&1
                local status=$?
            else 
                assertEquals 1 $status

                sleep 4

                $SEM /t trywait >> $Log 2>&1
                status=$?
                assertEquals 0 $status
            fi
        } &
    done

    wait
}

test008_multi_semaphores() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    $SEM /t0,/t1 open 2 > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    $SEM /t0 trywait >> $Log 2>&1
    status=$?
    assertEquals 0 $status

    $SEM /t0,/t1 getvalue >> $Log 2>&1

    $SEM /t0 post >> $Log 2>&1
    status=$?
    assertEquals 0 $status

    $SEM /t0,/t1 getvalue >> $Log 2>&1

    local ret=$(cat $Log | sed -ne "s/\[\/t[01]\].*\(OK value=[12]\)/\1/p" | tr -d "\n")
    assertEquals 'OK value=1OK value=2OK value=2OK value=2' "$ret"

    $SEM /t0,/t1 close >> $Log 2>&1
    status=$?
    assertEquals 0 $status

    $SEM /t0,/t1 unlink >> $Log 2>&1
    status=$?
    assertEquals 0 $status

}

# error test cases

test100_less_args() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    $SEM /t open > $Log 2>&1

    local ret=$(cat $Log | grep "'open' needs an initial value for semaphore." | wc -l)
    assertEquals 1 $ret
}

test101_not_open_and_trywait_error() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    $SEM /t trywait > $Log 2>&1

    local ret=$(cat $Log | grep "ERR=2; failed to open semaphore." | wc -l)
    assertEquals 1 $ret
}


# load and run shunit2
. $SHUNIT2_SRC_DIR/shunit2

