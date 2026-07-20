#!/bin/bash
# Diagnostic: check what kernel wait channels 109 threads are blocked on
for tid in $(ls /proc/15288/task/); do
    cat /proc/15288/task/$tid/wchan 2>/dev/null
    echo
done | sort | uniq -c | sort -rn
