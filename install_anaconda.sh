#!/usr/bin/bash

mkdir tmp
cd tmp
curl https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh --output anaconda.sh
sha256sum anaconda.sh
bash anaconda.sh -b -p ./../anaconda3
cd ..
rm -r tmp
