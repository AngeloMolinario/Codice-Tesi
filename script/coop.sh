#!/bin/bash

python3 coop_train.py config/coop/PE_joint.json
python3 coop_train.py config/coop/PE_coop.json
python3 coop_train.py config/coop/PE_vpt.json

python3 coop_train.py config/coop/SL_joint.json
python3 coop_train.py config/coop/SL_coop.json
python3 coop_train.py config/coop/SL_vpt.json