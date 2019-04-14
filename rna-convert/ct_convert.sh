#!/bin/bash

for file in `ls /mnt/e/all_ct_files/*`
do
    python main.py $file << EOF
2
EOF
done

cd /mnt/e/all_ct_files
rm -r *.ct

for file in `ls /mnt/e/all_ct_files/*`
do
    mv $file ${file%.ct(bpseq)}.bpseq
done
