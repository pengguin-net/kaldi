#!/usr/bin/env bash

nj=1

set -e

. ./cmd.sh
. ./path.sh
. parse_options.sh

librispeech_eg=../../librispeech/s5
model=$librispeech_eg/exp/chain_cleaned/tdnn_1d_sp
ivector_extractor=$librispeech_eg/exp/nnet3_cleaned/extractor
lang=$librispeech_eg/data/lang_test_tgsmall

utils/utt2spk_to_spk2utt.pl data/infer/utt2spk > data/infer/spk2utt


steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
    --cmd "$cmd" data/infer || exit 1;
steps/compute_cmvn_stats.sh data/infer || exit 1;
utils/fix_data_dir.sh data/infer

steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj $nj \
      data/infer $ivector_extractor data/infer/ivectors || exit 1;

steps/nnet3/compute_output.sh --cmd "$cmd" --nj $nj \
      --online-ivector-dir data/infer/ivectors data/infer $model exp/probs_infer

local/prepare_dict.sh data/local/lexicon.txt data/local/dict_nosp

utils/prepare_lang.sh --phone-symbol-table $lang/phones.txt \
  data/local/dict_nosp "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

utils/split_data.sh data/infer $nj
for i in `seq 1 $nj`; do
  utils/sym2int.pl -f 2- data/lang_nosp/words.txt \
    data/infer/split${nj}/$i/text \
    > data/infer/split${nj}/$i/text.int
done

utils/sym2int.pl -f 2- data/lang_nosp/phones.txt \
      data/local/text-phone > data/local/text-phone.int

$cmd JOB=1:$nj exp/ali_infer/log/mk_align_graph.JOB.log \
  compile-train-graphs-without-lexicon \
    --read-disambig-syms=data/lang_nosp/phones/disambig.int \
    $model/tree $model/final.mdl \
    "ark,t:data/infer/split${nj}/JOB/text.int" \
    "ark,t:data/local/text-phone.int" \
    "ark:|gzip -c > exp/ali_infer/fsts.JOB.gz"   || exit 1;

steps/align_mapped.sh --cmd "$cmd" --nj $nj --graphs exp/ali_infer \
  data/infer exp/probs_infer $lang $model exp/ali_infer

local/remove_phone_markers.pl $lang/phones.txt \
  data/lang_nosp/phones-pure.txt data/lang_nosp/phone-to-pure-phone.int

$cmd JOB=1:$nj exp/ali_infer/log/ali_to_phones.JOB.log \
  ali-to-phones --per-frame=true $model/final.mdl \
    "ark,t:gunzip -c exp/ali_infer/ali.JOB.gz|" \
    "ark,t:|gzip -c >exp/ali_infer/ali-phone.JOB.gz"   || exit 1;

$cmd JOB=1:$nj exp/gop_infer/log/compute_gop.JOB.log \
compute-gop --phone-map=data/lang_nosp/phone-to-pure-phone.int \
  --skip-phones-string=0:1:2 \
  $model/final.mdl \
  "ark,t:gunzip -c exp/ali_infer/ali-phone.JOB.gz|" \
  "ark:exp/probs_infer/output.JOB.ark" \
  "ark,scp:exp/gop_infer/gop.JOB.ark,exp/gop_infer/gop.JOB.scp" \
  "ark,scp:exp/gop_infer/feat.JOB.ark,exp/gop_infer/feat.JOB.scp"   || exit 1;

cat exp/gop_infer/feat.*.scp > exp/gop_infer/feat.scp
cat exp/gop_infer/gop.*.scp > exp/gop_infer/gop.scp
