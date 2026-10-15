#!/usr/bin/env bash
# Run eval/haystack.py on a GCP Spot L4 VM (project agentic-ai-iuk-ipf).
#
#   scripts/gcp_haystack.sh up   NAME            # create VM, push code + data, install env
#   scripts/gcp_haystack.sh run  NAME MODEL...   # start haystack.py in the background (BATCH=128)
#   scripts/gcp_haystack.sh babysit NAME...      # every 2 min: restart preempted VMs, resume runs
#   scripts/gcp_haystack.sh log  NAME            # tail the run log
#   scripts/gcp_haystack.sh pull NAME            # copy runs/ and vectors/ back into data/haystack/
#   scripts/gcp_haystack.sh down NAME            # delete the VM
#
# HS=<haystack name> (default GeneDiseaseHaystack) and CANDS=<candidates
# parquet, relative to the haystack dir> (default gene_pmids.parquet) pick
# the data for up, run and pull, e.g.
#   HS=GeneDiseaseHaystackDev CANDS=runs/bm25-symbol_disease-k0.9-b0.75__gene.parquet \
#       scripts/gcp_haystack.sh run NAME MODEL...
#
# `up` tries each EU zone with L4s until one has Spot capacity; the other
# commands look the VM's zone up.
#
# Doc vectors are cached in chunks on the VM's disk and finished models are
# skipped, so after a Spot preemption a restart plus `run` resumes; `run`
# saves its model list (and batch size, HS, CANDS) on the VM for `babysit`.
set -euo pipefail

ZONES=$(gcloud compute accelerator-types list --filter="zone~europe" \
    --format="value(name,zone.basename())" | awk '$1 == "nvidia-l4" {print $2}' | sort)
# BIOEMBED_REPO lets a copy of this script (e.g. a long-running babysit,
# immune to edits) still find the repo.
REPO=${BIOEMBED_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
HS=${HS:-GeneDiseaseHaystack}
CANDS=${CANDS:-gene_pmids.parquet}
HSDIR=data/haystack/$HS
cmd=$1 name=$2
shift 2
zone() { gcloud compute instances list --filter="name=$name" --format="value(zone.basename())"; }
ssh() { gcloud compute ssh "$name" --zone "$ZONE" --command "$1"; }
[ "$cmd" = up ] || ZONE=$(zone)

case $cmd in
up)
    for ZONE in $ZONES; do
    gcloud compute instances create "$name" --zone "$ZONE" \
        --network bioembed --machine-type g2-standard-8 --accelerator type=nvidia-l4,count=1 \
        --provisioning-model SPOT --instance-termination-action STOP \
        --maintenance-policy TERMINATE --boot-disk-size 200GB \
        --image-family common-cu129-ubuntu-2204-nvidia-580 \
        --image-project deeplearning-platform-release \
        --metadata install-nvidia-driver=True && break
    done
    ZONE=$(zone)
    [ -n "$ZONE" ] || { echo "no EU zone has Spot L4 capacity" >&2; exit 1; }
    until ssh true 2>/dev/null; do sleep 10; done
    ssh "mkdir -p BioEmbed/eval BioEmbed/models BioEmbed/$HSDIR"
    tar -C "$REPO" -czf - eval/*.py eval/pyproject.toml models/bioembed-bge-small-gaf \
        models/bioembed-bge-base-gaf data/tasks/GeneDiseaseRetrieval \
        $HSDIR/queries.parquet $HSDIR/$CANDS $HSDIR/docs_gene.parquet |
        gcloud compute ssh "$name" --zone "$ZONE" --command "tar -C BioEmbed -xzf -"
    # The local eval env pins CPU torch; on the VM use PyPI's CUDA build.
    ssh "curl -LsSf https://astral.sh/uv/install.sh | sh && cd BioEmbed/eval &&
         sed -i '/^\[tool.uv.sources\]/,/^explicit/d' pyproject.toml &&
         ~/.local/bin/uv sync --python 3.12 && ~/.local/bin/uv run python -c 'import torch; assert torch.cuda.is_available()'"
    ;;
run)
    ssh "echo '${BATCH:-128} $HS $CANDS $*' > ~/job.txt"
    # The remote shell lingers after detaching; the job survives the timeout.
    timeout 60 gcloud compute ssh "$name" --zone "$ZONE" --command \
        "cd BioEmbed/eval && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
         setsid nohup sh -c '~/.local/bin/uv run python haystack.py --batch-size ${BATCH:-128} \
         --haystack $HS --candidates $CANDS --models $* && rm ~/job.txt' \
         < /dev/null >> ~/haystack.log 2>&1 &" || true
    ;;
babysit)
    vms=("$name" "$@")
    while true; do
        for name in "${vms[@]}"; do
            # Subshell: one failed gcloud call (e.g. a preemption mid-check)
            # must not kill the babysitter under set -e.
            (
            ZONE=$(zone)
            if [ "$(gcloud compute instances describe "$name" --zone "$ZONE" \
                    --format='value(status)')" = TERMINATED ]; then
                echo "$(date -u +%FT%TZ) $name preempted, restarting"
                gcloud compute instances start "$name" --zone "$ZONE" || exit 0
                until ssh true 2>/dev/null; do sleep 10; done
            fi
            ssh "pgrep -f '[p]ython haystack.py' >/dev/null || ! test -f ~/job.txt" 2>/dev/null && exit 0
            read -r batch hs cands models < <(ssh "cat ~/job.txt")
            echo "$(date -u +%FT%TZ) $name resuming"
            BATCH=$batch HS=$hs CANDS=$cands "$0" run "$name" $models
            ) || echo "$(date -u +%FT%TZ) $name check failed, retrying next round"
        done
        sleep 120
    done
    ;;
log)
    ssh "grep -v -i warn ~/haystack.log | tail -20"
    ;;
pull)
    mkdir -p "$REPO/$HSDIR/runs"
    gcloud compute scp --zone "$ZONE" "$name:BioEmbed/$HSDIR/runs/*.parquet" "$REPO/$HSDIR/runs/"
    gcloud compute scp --zone "$ZONE" --recurse "$name:BioEmbed/$HSDIR/vectors" "$REPO/$HSDIR/"
    ;;
down)
    gcloud compute instances delete "$name" --zone "$ZONE" --quiet
    ;;
esac
