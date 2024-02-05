# ChatCLR2024 Challenge Task 2: Target Speaker Lipreading (Baseline)



## General Model Training

- **Lexicon and data directory**

  For training, development, and test sets, we prepare data directories and the lexicon in the format expected by  [kaldi](http://kaldi-asr.org/doc/data_prep.html) respectively. Note that we choose [DaCiDian](https://github.com/aishell-foundation/DaCiDian.git) raw resource and convert it to kaldi lexicon format.

- **Language model**

  We segment MISP speech transcription for language model training by applying [DaCiDian](https://github.com/aishell-foundation/DaCiDian.git) as dict and [Jieba](https://github.com/fxsjy/jieba) open-source toolkit. For the language model, we choose a maximum entropy-based 3-gram model, which achieves the best perplexity, from n-gram(n=2,3,4) models trained on MISP speech transcripts with different smoothing algorithms and parameters sets. And the selected 3-gram model has 516600 unigrams, 432247 bigrams, and 915962 trigrams respectively.  Note that the temporary and final language models are stored in /data/srilm.

- **GMM-HMM model**
  
  The GMM-HMM state model is used to obtain frame-level state label for DNN-based lipreading model training. For features extraction, we extract 13-dimensional MFCC features plus 3-dimensional pitches by using near speech. As a start point for triphone models, a monophone model is trained.  Then a triphone model is trained using delta features on the whole dataset. In the third triphone model training process, an MLLT-based global transform is estimated iteratively on the top of LDA feature to extract independent speaker features. For the fourth triphone model, feature space maximum likelihood linear regression (fMLLR) with speaker adaptive training (SAT) is applied in the training.

- **DNN-HMM model**

  Based on the tied-triphone state alignments from GMM, DNN is configured and trained to replace GMM. The input features is the 96 × 96 × 3 (*W × H × C*) lip ROI.

## Target Speaker Model Training

Building upon a pretrained General Model, we employ a subset of the target speaker's data to perform finetuning, deriving a Target Speaker Model. The finetuning process adheres to the training steps outlined for the DNN-HMM framework. When the evaluation set is unavailable, the development dataset is partitioned in a 7:3 ratio. The larger portion is allocated for finetuning purposes, whereas the smaller segment is reserved for testing. This division is explicitly documented in data.


## Results

- **Development set**

| SpeakerID           | S026  | S138  | S139  | S140  | S286  | S287  | S288  | S289  | S393  | S394  | S426  | S427  |
| General Model       | 97.85 | 95.96 | 97.08 | 96.90 | 98.08 | 95.31 | 96.95 | 96.64 | 95.87 | 97.07 | 97.01 | 96.27 |
| Target Speaker Model| | | | | | | | | | | | |
- **Evaluation set**

waiting.....

## Quick start

- **Setting Local System Jobs**

```
# Setting local system jobs (local CPU - no external clusters)
export train_cmd=run.pl
export decode_cmd=run.pl
```

- **Setting  Paths**

```
--- path.sh ---
# Defining Kaldi root directory
export KALDI_ROOT=
# Setting paths to useful tools
export PATH=
# Enable SRILM
. $KALDI_ROOT/tools/env.sh
# Variable needed for proper data sorting
export LC_ALL=C

--- run_misp.sh ---
# Defining corpus directory
misp2021_corpus=
# Defining path to python interpreter
python_path = 
# the directory to host coordinate information used to crop ROI 
data_roi =
# dictionary directory 
dict_dir= 
```

- **Run Training**

```
./run_misp.sh 
# options:
		--stage      -1  change the number to start from different training stages
```

## Requirments

- **Kaldi**

- **Python Packages:**

  numpy

  cv2

  pytorch

  [jieba](https://github.com/fxsjy/jieba)

- **Other Tools:**

  SRILM

