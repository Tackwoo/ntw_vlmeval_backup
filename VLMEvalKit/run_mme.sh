DEVICE=3
MODEL="llava-1.5-7b"
DATASET="COCO_VAL"
MODE="all"
VERBOSE="--verbose"

CUDA_VISIBLE_DEVICES=$DEVICE python run.py --model $MODEL --data $DATASET --mode $MODE --work-dir output/mme/unskip_012_293031/5 --fixed_skip_layer 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 $VERBOSE

