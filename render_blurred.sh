dataset=mvi_selected$2
datadir=datasets/MVImgNet/${dataset}

echo GPU $1
echo SPLIT $2
echo SCALE $3
echo REPEAT $4


while read -r name; do
    echo $name

    DATA_DIR=${datadir}/${name}
    echo DATA $DATA_DIR

    CKPT_DIR=results/${dataset}/${name}

    if [ -d $CKPT_DIR ]; then
        CUDA_VISIBLE_DEVICES=$1 python render_blurred.py -m $CKPT_DIR -r 2 --output_path results/renders_tmp --repeats $4 --expname r2_s$3 --scale $3
        CUDA_VISIBLE_DEVICES=$1 python render_blurred.py -m $CKPT_DIR -r 3 --output_path results/renders_tmp --repeats $4 --expname r3_s$3 --scale $3
        CUDA_VISIBLE_DEVICES=$1 python render_blurred.py -m $CKPT_DIR -r 4 --output_path results/renders_tmp --repeats $4 --expname r4_s$3 --scale $3

    else
        echo Not trained
    fi

done < ${datadir}/shuffled_list.txt 
