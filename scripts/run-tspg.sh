#!/bin/bash
#PBS -N tspg
#PBS -l select=1:ncpus=2:ngpus=1:mem=64gb:gpu_model=p100,walltime=72:00:00
#PBS -j oe

# Terminate the script if an error occurs
set -e

# Hide some tensorflow warnings
export TF_CPP_MIN_LOG_LEVEL="3"

# This should be the directory where you cloned the TSPG repository
TSPG_DIR="${HOME}/TSPG"

# The input data should reside somewhere in your home or zfs directory,
# change these variables to match your specific data, gene set, target class
INPUT_DIR="input"
TRAIN_DATA="${INPUT_DIR}/breast_train_GEM_transpose.txt"
TRAIN_LABELS="${INPUT_DIR}/train_labels.txt"
PERTURB_DATA="${INPUT_DIR}/breast_test_GEM_transpose.txt"
PERTURB_LABELS="${INPUT_DIR}/test_labels_new.txt"
GMT_FILE="${INPUT_DIR}/breast_gene_list.txt"
GENE_SET="Gene_set_all"
TARGET_CLASS="normal"

# The output data will be written to local scratch ($TMPDIR) during
# the workflow, then copied to $OUTPUT_DIR at the end, to minimize
# I/O on your home / zfs directory
OUTPUT_DIR="output"

# Create conda environment from instructions in TSPG readme
module purge
#module load anaconda3/5.1.0-gcc

#source activate tspg

# train target model on a gene set
echo
echo "PHASE 1: TRAIN TARGET MODEL"
echo

${TSPG_DIR}/bin/train-target.py \
    --dataset        ${TRAIN_DATA} \
    --labels         ${TRAIN_LABELS} \
    --perturb-data   ${PERTURB_DATA}\
    --gene-sets      ${GMT_FILE} \
    --set            ${GENE_SET} \
    --output-dir     ${TSPG_DIR}/${OUTPUT_DIR}

# train AdvGAN model on a gene set
echo
echo "PHASE 2: TRAIN PERTURBATION GENERATOR"
echo

${TSPG_DIR}/bin/train-advgan.py \
    --dataset        ${TRAIN_DATA} \
    --labels         ${TRAIN_LABELS} \
    --perturb-data   ${PERTURB_DATA}\
    --gene-sets      ${GMT_FILE} \
    --set            ${GENE_SET} \
    --target         ${TARGET_CLASS} \
    --target-cov diagonal \
    --output-dir ${TSPG_DIR}/${OUTPUT_DIR}

# generate perturbed samples using AdvGAN model
echo
echo "PHASE 3: GENERATE SAMPLE PERTURBATIONS"
echo

${TSPG_DIR}/bin/perturb.py \
    --train-data      ${TRAIN_DATA} \
    --train-labels    ${TRAIN_LABELS} \
    --perturb-data    ${PERTURB_DATA} \
    --perturb-labels  ${PERTURB_LABELS} \
    --gene-sets       ${GMT_FILE} \
    --set             ${GENE_SET} \
    --target          ${TARGET_CLASS} \
    --output-dir      ${TSPG_DIR}/${OUTPUT_DIR}

# create t-SNE and heatmap visualizations of perturbed samples for a gene set
echo
echo "PHASE 4: VISUALIZE SAMPLE PERTURBATIONS"
echo

${TSPG_DIR}/bin/visualize.py \
    --train-data      ${TRAIN_DATA} \
    --train-labels    ${TRAIN_LABELS} \
    --perturb-data    ${PERTURB_DATA} \
    --perturb-labels  ${PERTURB_LABELS} \
    --gene-sets       ${GMT_FILE} \
    --set             ${GENE_SET} \
    --target          ${TARGET_CLASS} \
    --output-dir      ${TSPG_DIR}/${OUTPUT_DIR} \
    --tsne \
    --umap \
    --tsne-npca 50 
#--heatmap 

# save output data to permanent storage
#rm -rf ${OUTPUT_DIR}
#cp -r ${TSPG_DIR}/${OUTPUT_DIR} .
