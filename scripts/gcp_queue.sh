#!/usr/bin/env bash
# Unattended job queues on on-demand GCP L4 VMs (project agentic-ai-iuk-ipf).
# Dan's gcloud login lasts about an hour, so each VM runs its whole queue and
# powers itself off without this script (VM side: scripts/gcp_runner.sh).
#
#   scripts/gcp_queue.sh up     NAME [PATH...]  # create VM, push code + data (+ PATHs), install env + runner
#   scripts/gcp_queue.sh queue  NAME FILE       # append FILE's lines (one shell command each) to the queue
#   scripts/gcp_queue.sh status NAME            # job states, current log tail, GPU
#   scripts/gcp_queue.sh pull   NAME|all        # copy back every file the jobs wrote, and the queue logs
#   scripts/gcp_queue.sh down   NAME            # delete the VM and its disk
#
# Jobs run from ~/BioEmbed/eval on the VM, e.g.
#   uv run python haystack.py --haystack GeneDiseaseHaystackDev --batch-size 512 --models BAAI/bge-small-en-v1.5
#
# up: g2-standard-8 + 1 L4, on-demand, 300 GB boot disk; tries each EU zone
# with L4s in turn (ZONE=... to pin one), ROUNDS times (default 1) a minute
# apart. Pushes eval/ code (+ uv.lock), tests/, scripts/,
# data/tasks/GeneDiseaseRetrieval and, for each haystack in HAYSTACKS
# (default: practice and test), queries, gene_pmids and docs_gene; PATHs
# (repo-relative) are pushed too. JOBS=FILE queues FILE before the data
# goes up, so the VM starts on its own once setup ends, even if the login
# expires mid-push. The push and the setup share one ssh session. A boot-time
# guard powers the VM off 1 h after boot if the runner is not active and
# nothing is being pushed.
#
# A stopped VM may not start again (no free L4 in its zone), so results never
# wait on it: after each job the VM syncs every file the jobs wrote to
# gs://$BUCKET/NAME/files/ and its queue (jobs, state, logs) to .../queue/.
# pull copies those into the same paths here (DEST=dir to land them
# elsewhere; a newer local copy is kept) and the queue to
# data/gcp_logs/NAME/; it works while the VM runs, after it powered off, or
# after down. 'pull all' does every VM in the bucket, e.g. hs-ft's models/
# and data/finetune_gd/runs/, hs-pm's genesearch/index/emb_medembed_small/.
# status reads GCS when the VM is off. queue starts a stopped VM.
set -euo pipefail
shopt -s inherit_errexit

REPO=${BIOEMBED_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
DEST=${DEST:-$REPO}
HAYSTACKS=${HAYSTACKS-GeneDiseaseHaystackDev GeneDiseaseHaystack}
BUCKET=${BUCKET:-agentic-ai-iuk-ipf-bioembed-queue}  # created once: gcloud storage buckets create gs://$BUCKET --location=EU
cmd=${1:?usage: $0 up|queue|status|pull|down NAME ...} name=${2:?NAME required}
shift 2

zone() { gcloud compute instances list --filter="name=$name" --format="value(zone.basename())"; }
vm_status() { gcloud compute instances describe "$name" --zone "$ZONE" --format="value(status)"; }
ssh() { gcloud compute ssh "$name" --zone "$ZONE" --command "$1"; }
wait_ssh() {
    for _ in $(seq 30); do timeout 60 gcloud compute ssh "$name" --zone "$ZONE" --command true \
        2>/dev/null && return; sleep 10; done
    echo "$name: no ssh after 30 tries" >&2
    exit 1
}
start() {  # start a stopped VM (fails if its zone has no free L4)
    while [ "$(vm_status)" = STOPPING ]; do sleep 10; done
    [ "$(vm_status)" = RUNNING ] || gcloud compute instances start "$name" --zone "$ZONE"
    wait_ssh
}

if [ "$cmd" = queue ] || [ "$cmd" = status ] || [ "$cmd" = down ]; then
    ZONE=$(zone)
    [ -n "$ZONE" ] || { echo "no VM named $name" >&2; exit 1; }
fi

case $cmd in
up)
    paths=(eval/pyproject.toml eval/uv.lock tests scripts data/tasks/GeneDiseaseRetrieval "$@")
    for f in "$REPO"/eval/*.py; do paths+=("eval/${f##*/}"); done
    for hs in $HAYSTACKS; do
        paths+=(data/haystack/$hs/queries.parquet data/haystack/$hs/gene_pmids.parquet data/haystack/$hs/docs_gene.parquet)
    done
    for p in "${paths[@]}"; do [ -e "$REPO/$p" ] || { echo "missing $p" >&2; exit 1; }; done
    du -shc "${paths[@]/#/$REPO/}" | tail -n 1 | sed 's/total/to push/'
    guard=$(mktemp)
    cat > "$guard" <<'EOF'
#!/bin/sh
systemd-run --on-active=1h /bin/sh -c 'while pgrep -x tar >/dev/null; do sleep 300; done; systemctl is-active --quiet bioembed-queue || poweroff'
EOF
    zones=${ZONE:-$(gcloud compute accelerator-types list --filter="zone~europe" \
        --format="value(name,zone.basename())" | awk '$1 == "nvidia-l4" {print $2}' | sort)}
    for round in $(seq "${ROUNDS:-1}"); do
        for ZONE in $zones; do
            gcloud compute instances create "$name" --zone "$ZONE" \
                --network bioembed --machine-type g2-standard-8 --accelerator type=nvidia-l4,count=1 \
                --provisioning-model STANDARD --maintenance-policy TERMINATE --restart-on-failure \
                --boot-disk-size 300GB --image-family common-cu129-ubuntu-2204-nvidia-580 \
                --image-project deeplearning-platform-release \
                --scopes cloud-platform --metadata install-nvidia-driver=True,bioembed-bucket="$BUCKET" \
                --metadata-from-file startup-script="$guard" \
                2> >(grep -E '^(ERROR|code:| - )' | grep -v 'Disk size' >&2) && break 2
        done
        echo "round $round: no capacity in $(echo $zones | wc -w) zones" >&2
        sleep 60
    done
    rm -f "$guard"
    ZONE=$(zone)
    [ -n "$ZONE" ] || { echo "no EU zone has an on-demand L4 free" >&2; exit 1; }
    echo "$name: $ZONE"
    wait_ssh
    if [ -n "${JOBS:-}" ]; then
        sed -e '$a\' "$JOBS" | ssh "mkdir -p queue && cat >> queue/jobs.txt"
    fi
    tar -C "$REPO" --exclude __pycache__ -cf - "${paths[@]}" |
        ssh "mkdir -p BioEmbed && tar -C BioEmbed -xf - && bash BioEmbed/scripts/gcp_runner.sh setup"
    ;;
queue)
    file=${1:?FILE required}
    start
    sed -e '$a\' "$file" | ssh "cat >> queue/jobs.txt && echo 0 > queue/hold"
    ssh "bash BioEmbed/scripts/gcp_runner.sh status"
    ;;
status)
    st=$(vm_status)
    echo "$name ($ZONE): $st"
    if [ "$st" = RUNNING ]; then
        ssh "bash BioEmbed/scripts/gcp_runner.sh status"
    else
        tmp=$(mktemp -d)
        gcloud storage rsync -r --no-user-output-enabled "gs://$BUCKET/$name/queue" "$tmp"
        echo "powered off (queue ran out, or stopped by hand); as last synced:"
        Q=$tmp bash "$REPO/scripts/gcp_runner.sh" jobs
        rm -rf "$tmp"
    fi
    ;;
pull)
    [ "$name" = all ] && names=$(gcloud storage ls "gs://$BUCKET/" | sed 's|.*/\([^/]*\)/$|\1|') || names=$name
    for name in $names; do
        echo "=== $name"
        mkdir -p "$DEST/data/gcp_logs/$name"
        gcloud storage rsync -r -u "gs://$BUCKET/$name/files" "$DEST"
        gcloud storage rsync -r "gs://$BUCKET/$name/queue" "$DEST/data/gcp_logs/$name"
    done
    ;;
down)
    gcloud compute instances delete "$name" --zone "$ZONE" --quiet
    ;;
*)
    echo "unknown command $cmd" >&2
    exit 2
    ;;
esac
