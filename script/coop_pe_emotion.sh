#!/bin/bash

echo "#######################################################################"
echo "##                  TRAINING EMOTION ON COOP 15                      ##"
echo "#######################################################################"
python3 coop_train.py config/coop/emotion/PE_coop_15.json

echo "#######################################################################"
echo "##                  TRAINING EMOTION ON COOP 15                      ##"
echo "#######################################################################"
python3 coop_train.py config/coop/emotion/PE_coop_20.json

echo "#######################################################################"
echo "##                  TRAINING EMOTION ON COOP 25                      ##"
echo "#######################################################################"
python3 coop_train.py config/coop/emotion/PE_coop_25.json


echo "#######################################################################"
echo "##                  TRAINING EMOTION ON VPT 10                       ##"
echo "#######################################################################"
python3 coop_train.py config/coop/emotion/PE_vpt_10.json

echo "#######################################################################"
echo "##                  TRAINING EMOTION ON VPT 20                       ##"
echo "#######################################################################"
python3 coop_train.py config/coop/emotion/PE_vpt_20.json

echo "#######################################################################"
echo "##                  TRAINING EMOTION ON VPT 30                       ##"
echo "#######################################################################"
python3 coop_train.py config/coop/emotion/PE_vpt_30.json
