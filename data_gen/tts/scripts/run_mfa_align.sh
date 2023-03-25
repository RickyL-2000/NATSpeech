#!/bin/bash
set -e

NUM_JOB=${NUM_JOB:-36}
echo "| Aligning MFA using ${NUM_JOB} cores."
BASE_DIR=data/processed/$CORPUS
rm -rf $BASE_DIR/mfa_outputs_tmp
mfa train $BASE_DIR/mfa_inputs $BASE_DIR/mfa_dict.txt $MFA_MODEL_DIR/mfa_model.zip $BASE_DIR/mfa_outputs_tmp -t $BASE_DIR/mfa_tmp --clean -j $NUM_JOB
rm -rf $BASE_DIR/mfa_tmp $BASE_DIR/mfa_outputs
mkdir -p $BASE_DIR/mfa_outputs
find $BASE_DIR/mfa_outputs_tmp -maxdepth 1 -regex ".*/[0-9]+" -print0 | xargs -0 -i rsync -a {}/ $BASE_DIR/mfa_outputs/
rm -rf $BASE_DIR/mfa_outputs_tmp
