#!/bin/bash

inotifywait -e close_write,moved_to,create -m . |
while read -r directory events filename; do
    if [ "${filename: -4}" = ".tex" ]; then
        ./compile_user_manual.sh & evince user_manual.pdf
    fi
done
