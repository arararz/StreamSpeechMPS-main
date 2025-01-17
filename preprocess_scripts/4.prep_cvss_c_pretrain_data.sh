lang=$1
CVSS_ROOT=/Users/arararz/Documents/datasets/cvss/cvss-c
ROOT=/Users/arararz/Documents/GitHub/StreamSpeech

PYTHONPATH=$ROOT/fairseq python $ROOT/preprocess_scripts/convert_s2st_tsv_to_s2tt_tsv.py \
    --s2st-tsv-dir $CVSS_ROOT/${lang}-en/fbank2unit \
    --s2tt-tsv-dir $CVSS_ROOT/${lang}-en/fbank2text 

cp $CVSS_ROOT/${lang}-en/tgt_unigram6000/spm* $CVSS_ROOT/${lang}-en/fbank2text 