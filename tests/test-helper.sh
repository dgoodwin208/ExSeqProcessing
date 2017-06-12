#!/bin/bash

get_value_by_key() {
    local key="$2"
    local value=$(sed -ne "s/.*$key *: *\(.*\)$/\1/p" "$1")
    echo $value
}

get_key_by_value() {
    local value="$2"
    local key=$(sed -ne "s/ *\([^ ]*\) *: *$value/\1/p" "$1")
    echo $key
}


