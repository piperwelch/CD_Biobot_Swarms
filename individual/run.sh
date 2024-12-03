#!/bin/bash

BOT_ID="Run2group9subject1"
RES="50"

START=0
STOP=499

SAVE_VOXELS=0 # Always 0 for batch simulations

# ---------- TREATMENT ----------
CTRL_RANDOMIZE_CILIA=0
SPHERE=0 
RESTRICT=1
TX='tx'

FILENAME=$BOT_ID"_res"$RES
SAVE_DIR_TX=$BOT_ID"_res"$RES"_"$TX
if [ ! -d "output_data/"$SAVE_DIR_TX ] 
then
    mkdir "output_data/"$SAVE_DIR_TX
    echo "output_data/"$SAVE_DIR_TX "created"
else
    echo "output_data/"$SAVE_DIR_TX "already exists"
fi

# Set up and submit sims
for SEED in `seq $START $STOP`
do
    python main.py $SEED $FILENAME $SAVE_VOXELS $CTRL_RANDOMIZE_CILIA $SPHERE $RESTRICT $TX
    sbatch --job-name="tx" --export=BOT_ID=$FILENAME,SEED=$SEED,SAVE_DIR=$SAVE_DIR_TX submit.sh
done

# ---------- CONTROL 1 ----------
CTRL_RANDOMIZE_CILIA=1 # control 1
SPHERE=1 # control 1
RESTRICT=1
CTRL='ctrl'

FILENAME=$BOT_ID"_res"$RES
SAVE_DIR_CTRL=$BOT_ID"_res"$RES"_ctrl"
if [ ! -d "output_data/"$SAVE_DIR_CTRL ] 
then
    mkdir "output_data/"$SAVE_DIR_CTRL
    echo "output_data/"$SAVE_DIR_CTRL "created"
else
    echo "output_data/"$SAVE_DIR_CTRL "already exists"
fi

# Set up and submit sims
for SEED in `seq $START $STOP`
do
    python main.py $SEED $FILENAME $SAVE_VOXELS $CTRL_RANDOMIZE_CILIA $SPHERE $RESTRICT $CTRL
    sbatch --job-name="ctrl" --export=BOT_ID=$FILENAME,SEED=$SEED,SAVE_DIR=$SAVE_DIR_CTRL submit.sh
done

# ---------- CONTROL 2 ----------
CTRL_RANDOMIZE_CILIA=1 
SPHERE=0 # control 2
RESTRICT=1
CTRL2='ctrl2'

FILENAME=$BOT_ID"_res"$RES
SAVE_DIR_CTRL2=$BOT_ID"_res"$RES"_ctrl2"
if [ ! -d "output_data/"$SAVE_DIR_CTRL2 ] 
then
    mkdir "output_data/"$SAVE_DIR_CTRL2
    echo "output_data/"$SAVE_DIR_CTRL2 "created"
else
    echo "output_data/"$SAVE_DIR_CTRL2 "already exists"
fi

# Set up and submit sims
for SEED in `seq $START $STOP`
do
    python main.py $SEED $FILENAME $SAVE_VOXELS $CTRL_RANDOMIZE_CILIA $SPHERE $RESTRICT $CTRL2
    sbatch --job-name="ctrl2" --export=BOT_ID=$FILENAME,SEED=$SEED,SAVE_DIR=$SAVE_DIR_CTRL2 submit.sh
done

# ---------- CONTROL 3 ----------
CTRL_RANDOMIZE_CILIA=0 # must be 0
SPHERE=1 # control 3
RESTRICT=1
CTRL3='ctrl3'

FILENAME=$BOT_ID"_res"$RES
SAVE_DIR_CTRL3=$BOT_ID"_res"$RES"_ctrl3"
if [ ! -d "output_data/"$SAVE_DIR_CTRL3 ] 
then
    mkdir "output_data/"$SAVE_DIR_CTRL3
    echo "output_data/"$SAVE_DIR_CTRL3 "created"
else
    echo "output_data/"$SAVE_DIR_CTRL3 "already exists"
fi

# Set up and submit sims
for SEED in `seq $START $STOP`
do
    # FILE=data/${SAVE_DIR_CTRL3}_run${SEED}    
    # if [ -d $FILE ]; then
    #     echo "File for run $SEED exists."
    # else
    #     echo "File for run $SEED does not exist."
    #     python main.py $SEED $FILENAME $SAVE_VOXELS $CTRL_RANDOMIZE_CILIA $SPHERE $RESTRICT $CTRL3
    #     sbatch --job-name="ctrl3" --export=BOT_ID=$FILENAME,SEED=$SEED,SAVE_DIR=$SAVE_DIR_CTRL3 submit.sh
    # fi
    python main.py $SEED $FILENAME $SAVE_VOXELS $CTRL_RANDOMIZE_CILIA $SPHERE $RESTRICT $CTRL3
    sbatch --job-name="ctrl3" --export=BOT_ID=$FILENAME,SEED=$SEED,SAVE_DIR=$SAVE_DIR_CTRL3 submit.sh
done

# ---------- CONTROL 4 ----------
CTRL_RANDOMIZE_CILIA=0
SPHERE=0 
RESTRICT=0 # control 4
CTRL4='ctrl4'

FILENAME=$BOT_ID"_res"$RES
SAVE_DIR_CTRL4=$BOT_ID"_res"$RES"_"$CTRL4
if [ ! -d "output_data/"$SAVE_DIR_CTRL4 ] 
then
    mkdir "output_data/"$SAVE_DIR_CTRL4
    echo "output_data/"$SAVE_DIR_CTRL4 "created"
else
    echo "output_data/"$SAVE_DIR_CTRL4 "already exists"
fi

# Set up and submit sims
for SEED in `seq $START $STOP`
do

    # FILE=data/${SAVE_DIR_CTRL4}_run${SEED}/body_${SAVE_DIR_CTRL4}_run${SEED}.vxd
    # if [ -f $FILE ]; then
    #     echo "File for run $SEED exists."
    # else
    #     echo "File for run $SEED does not exist."
    #     python main.py $SEED $FILENAME $SAVE_VOXELS $CTRL_RANDOMIZE_CILIA $SPHERE $RESTRICT $CTRL4
    #     sbatch --job-name="ctrl4" --export=BOT_ID=$FILENAME,SEED=$SEED,SAVE_DIR=$SAVE_DIR_CTRL4 submit.sh
    # fi

    python main.py $SEED $FILENAME $SAVE_VOXELS $CTRL_RANDOMIZE_CILIA $SPHERE $RESTRICT $CTRL4
    sbatch --job-name="ctrl4" --export=BOT_ID=$FILENAME,SEED=$SEED,SAVE_DIR=$SAVE_DIR_CTRL4 submit.sh
done
