#!/bin/bash

cd weights

# Target-conditional X
gdown https://drive.google.com/uc?id=1romQcMrJ7UM6Ps7sFrrvGp4Rmj_I2OBi

# Target-conditional O
gdown https://drive.google.com/uc?id=13ATBmJIgIhU3Kclwut0yssjaciPCieEV

cd ../data/building_blocks

# zincfrag.smi.gz
gdown https://drive.google.com/uc?id=16N8Xyxr9a-CifjIofgdH3ssFukC4Eh_V

# zincfrag_5.6M.smi.gz
gdown https://drive.google.com/uc?id=1GKrf-ewTJkZ6SchO186fBDl_IQqieTei

# gunzip *.smi.gz

#LIT-PCBA.tar.gz
gdown https://drive.google.com/uc?id=1A93RIkhuXNQyWCnqSTPfvhHlgaWMnHx5

# CrossDocked2020_test.tar.gz
gdown https://drive.google.com/uc?id=1A5LVjruLHczpfvA4e5JnMUgpbKRvejfJ

# CrossDocked2020_all.tar.gz
gdown https://drive.google.com/uc?id=1iGr053FDC9tCYz4es4cRJ6WpkEEi3CAW

# for archive in *.tar.gz; do
# tar -xzvf "$archive"
# done