#!/usr/bin/env bash
# Model check: embed one shard of the subset (subset_texts.py) with MODEL on a GCP Spot L4, end to end.
#   geometry/manifold/gcp_embed.sh NAME SHARD [MODEL]
# Creates VM NAME in the first EU zone with Spot L4 capacity, pushes gpu_embed.py and the shard's texts, installs
# sentence-transformers (CUDA torch from PyPI), embeds papers_SHARD then concepts_SHARD (resumable chunks),
# restarts the VM and the job after a preemption, pulls the vectors to data/geometry/manifold_check/<model>/,
# then deletes the VM. Uses the iuk-pipelines service account (Dan's own gcloud token expires daily).
set -uo pipefail
export CLOUDSDK_CORE_ACCOUNT=iuk-pipelines@agentic-ai-iuk-ipf.iam.gserviceaccount.com CLOUDSDK_CORE_PROJECT=agentic-ai-iuk-ipf
name=$1 shard=$2 model=${3:-Qwen/Qwen3-Embedding-8B}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=$REPO/data/geometry/manifold_check
OUT=$DATA/$(basename "$model"); mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $name: $*"; }
zone() { gcloud compute instances list --filter="name=$name" --format="value(zone.basename())"; }
ssh() { gcloud compute ssh "$name" --zone "$ZONE" --command "$1"; }

ZONES=$(gcloud compute accelerator-types list --filter="zone~europe" --format="value(name,zone.basename())" |
        awk '$1 == "nvidia-l4" {print $2}' | sort)
ZONE=$(zone)
until [ -n "$ZONE" ]; do
    for z in $ZONES; do
        gcloud compute instances create "$name" --zone "$z" --network bioembed --machine-type g2-standard-8 \
            --accelerator type=nvidia-l4,count=1 --provisioning-model SPOT --instance-termination-action STOP \
            --maintenance-policy TERMINATE --boot-disk-size 150GB --image-family common-cu129-ubuntu-2204-nvidia-580 \
            --image-project deeplearning-platform-release --metadata install-nvidia-driver=True >/dev/null 2>&1 \
            && { log "created in $z"; break; }
    done
    ZONE=$(zone); [ -n "$ZONE" ] || { log "no Spot L4 capacity in any EU zone; retrying in 5 min"; sleep 300; }
done
until ssh true 2>/dev/null; do sleep 10; done
if ! ssh "test -x ~/st/bin/python" 2>/dev/null; then
    log "installing"
    gcloud compute scp --zone "$ZONE" "$REPO/geometry/manifold/gpu_embed.py" "$DATA/papers_$shard.parquet" \
        "$DATA/concepts_$shard.parquet" "$name:" >/dev/null
    ssh "curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null && ~/.local/bin/uv venv -q --python 3.12 ~/st &&
         VIRTUAL_ENV=~/st ~/.local/bin/uv pip install -q sentence-transformers torch pandas pyarrow &&
         ~/st/bin/python -c 'import torch; assert torch.cuda.is_available()'" || { log "install failed"; exit 1; }
fi
start_job() {
    timeout 60 gcloud compute ssh "$name" --zone "$ZONE" --command \
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup sh -c '~/st/bin/python gpu_embed.py $model papers_$shard.parquet out &&
         ~/st/bin/python gpu_embed.py $model concepts_$shard.parquet out --batch 128 && touch out/ALL_DONE' < /dev/null >> embed.log 2>&1 &" || true
}
log "starting job"; start_job
while true; do
    sleep 120
    st=$(gcloud compute instances describe "$name" --zone "$ZONE" --format='value(status)' 2>/dev/null)
    if [ "$st" = TERMINATED ]; then
        log "preempted, restarting"
        until gcloud compute instances start "$name" --zone "$ZONE" >/dev/null 2>&1; do sleep 120; done
        until ssh true 2>/dev/null; do sleep 10; done
        start_job; continue
    fi
    if ssh "test -f out/ALL_DONE" 2>/dev/null; then break; fi
    ssh "pgrep -f '[g]pu_embed.py' >/dev/null" 2>/dev/null || { log "job not running, restarting it"; ssh "tail -3 embed.log"; start_job; }
    ssh "grep -v -i warn embed.log | tail -1" 2>/dev/null | sed "s/^/$name: /"
done
log "pulling"
gcloud compute scp --zone "$ZONE" "$name:out/papers_$shard.npy" "$name:out/concepts_$shard.npy" "$OUT/" && \
    gcloud compute instances delete "$name" --zone "$ZONE" --quiet && log "done, VM deleted"
