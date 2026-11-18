#!/bin/bash
# Host A evaluation chain for one arm's run dir. usage: eval_all.sh RUN_DIR ARM SPLIT "COND_LIST" "PARA_CONDS"
set -u
RUN=$1; ARM=$2; SPLIT=$3; CONDS=$4; PARA=$5
S=~/Documents/langres/scripts; PY="nice -n 10 $HOME/langres_state/venv/bin/python"
cd ~/langres_state
t() { echo "$(date -u +%FT%TZ) $*"; }
t "page-cache drop not permitted (LR-X15); evicting own files"; $PY $S/evict_cache.py ~/langres_data/extracted >/dev/null
for c in $PARA; do t "paraphrase $c"; $PY $S/eval_lm.py --task para --run $RUN --cond $c --split $SPLIT || t "FAIL para $c"
  t "rescore $c"; $PY $S/rescore.py --run $RUN --arm $ARM --infile $RUN/${c}_para_final.jsonl || t "FAIL rescore $c"; done
FILES=""; for c in $CONDS; do FILES="$FILES $RUN/${c}_final.jsonl"; done
for c in $PARA; do FILES="$FILES $RUN/${c}_para_final.jsonl"; done
t "check scorer"; $PY $S/check_scorer.py score --arm $ARM --split $SPLIT --run_dir $RUN --files $FILES || t "FAIL check"
for c in $CONDS; do t "ppl $c"; $PY $S/eval_lm.py --task ppl --run $RUN --cond $c --split $SPLIT || t "FAIL ppl $c"; done
for c in $CONDS; do t "qa $c"; $PY $S/eval_lm.py --task qa --run $RUN --cond $c --split $SPLIT || t "FAIL qa $c"; done
t "readouts"; $PY $S/readouts.py $RUN $SPLIT --conds $CONDS || t "FAIL readouts"
t EVAL_ALL_DONE
