#!/bin/bash
#SBATCH --job-name=audio_seq_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=audio_seq_v2_%j.out
#SBATCH --error=audio_seq_v2_%j.err

cd /project2/msoleyma_1026/CalmToConflict/andy/hubert_audio_andy

source ~/.bashrc
conda activate icformer

python -u extract_audio_sequences_v2.py