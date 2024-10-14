dataset=mvi_selected$2
datadir=datasets/MVImgNet/${dataset}

echo GPU $1
echo SPLIT $2
echo PORT $3

while read -r name; do
    echo $name

    DATA_DIR=${datadir}/${name}
    echo DATA $DATA_DIR

    CKPT_DIR=results/${dataset}/${name}
    CHECK_DONE=results/${dataset}/${name}/Done_train

    if [ -d $CKPT_DIR ]; then
        echo 'Training Done'
    else
        mkdir -p $CKPT_DIR    
        echo CKPT $CKPT_DIR
        CUDA_VISIBLE_DEVICES=$1 python train.py -s $DATA_DIR -m $CKPT_DIR --port $3 #--eval
    fi
    
done < ${datadir}/shuffled_list.txt 
