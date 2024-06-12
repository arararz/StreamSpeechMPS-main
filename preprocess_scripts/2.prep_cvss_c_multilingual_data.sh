lang=$1
CVSS_ROOT=/Users/arararz/Documents/datasets/cvss/cvss-c
COVOST2_ROOT=/Users/arararz/Documents/datasets/covost2
ROOT=/Users/arararz/Documents/GitHub/StreamSpeech
PRETRAIN_ROOT=$ROOT/pretrain_models

PYTHONPATH=$ROOT/fairseq python $ROOT/preprocess_scripts/prep_cvss_c_multilingual_data.py \
    --covost-data-root $COVOST2_ROOT/ --cvss-data-root $CVSS_ROOT/ \
    --output-root $CVSS_ROOT/$lang-en \
    --src-lang $lang \
    --target-type unit --unit-type km1000 --reduce-unit \
    --vocoder-checkpoint $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000 --vocoder-cfg $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json

