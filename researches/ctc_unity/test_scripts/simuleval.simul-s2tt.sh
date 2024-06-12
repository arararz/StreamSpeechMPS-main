export mps_VISIBLE_DEVICES=0

ROOT=/Users/arararz/Documents/GitHub/StreamSpeech
DATA_ROOT=/Users/arararz/Documents/datasets/cvss/cvss-c

LANG=fr
file=/Users/arararz/Documents/GitHub/StreamSpeech_model/streamspeech.simultaneous.${LANG}-en.pt
output_dir=$ROOT/res/streamspeech.simultaneous.${LANG}-en/simul-s2tt

chunk_size=960

PYTHONPATH=$ROOT/fairseq simuleval --data-bin ${DATA_ROOT}/${LANG}-en/fbank2unit \
    --user-dir ${ROOT}/agent \
    --source ${DATA_ROOT}/${LANG}-en/simuleval/test/wav_list.txt --target ${DATA_ROOT}/${LANG}-en/simuleval/test/target.txt \
    --model-path $file \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_unity_asr_st_ctcst.yaml \
    --agent $ROOT/agent/speech_to_text.s2tt.streamspeech.agent.py\
    --output $output_dir/chunk_size=$chunk_size \
    --source-segment-size $chunk_size \
    --quality-metrics BLEU --latency-metrics AL AP DAL StartOffset EndOffset LAAL ATD NumChunks RTF \
    --device gpu --computation-aware 