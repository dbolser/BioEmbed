#!/usr/bin/env bash
# VM side of scripts/gcp_queue.sh: an unattended job queue that powers the VM
# off when it is empty.
#
#   gcp_runner.sh setup    # once, from `up`: service, eval env with CUDA torch, then ~/queue/ready
#   gcp_runner.sh run      # the systemd service (bioembed-queue)
#   gcp_runner.sh status   # job states, current log tail, GPU
#   Q=DIR gcp_runner.sh jobs   # job states only, e.g. of a queue copied from GCS
#
# ~/queue/jobs.txt: one shell command per line, run by bash -o pipefail from
# ~/BioEmbed/eval; blank and '#' lines are skipped. Job N is line N. Exit 0
# writes state/N.done; anything else writes state/N.failed (exit code + log
# tail) and the queue moves on. Output goes to logs/N.log. A job cut off by a
# reboot runs again (haystack.py and embed_corpus.py resume), at most TRIES
# starts. Jobs wait for ~/queue/ready.
#
# After each job, and before powering off, every file the jobs wrote under
# ~/BioEmbed/{data,models,genesearch} (newer than ready; via hard links in
# ~/outbox; symlinks are skipped) goes to gs://<bucket>/<vm>/files/ and ~/queue to
# gs://<bucket>/<vm>/queue/ (bucket: instance metadata bioembed-bucket), so
# results never depend on restarting a VM in a zone with no free L4.
#
# With no job left the VM powers itself off, unless ~/queue/hold (an epoch
# time, written this boot by up/queue) is in the future. With no hold
# written this boot it waits until GRACE seconds of uptime, so `queue` can
# reach a VM it has just started.
set -uo pipefail

Q=${Q:-$HOME/queue}
TRIES=3
GRACE=600
UP_HOLD=21600  # setup keeps the VM up 6 h for `queue` if no jobs came with `up`
UV=$HOME/.local/bin/uv

log() { echo "$(date -u +%FT%TZ) $*" >> "$Q/runner.log"; }

meta() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }

sync() {
    local out=$HOME/outbox dest
    dest=gs://$(meta attributes/bioembed-bucket)/$(meta name)
    (cd "$HOME/BioEmbed" && find $(ls -d data models genesearch 2>/dev/null) -newer "$Q/ready" \
        -type f ! -name '*.tmp*') | while read -r f; do
        mkdir -p "$out/$(dirname "$f")"
        ln -f "$HOME/BioEmbed/$f" "$out/$f"
    done
    mkdir -p "$out"
    if gcloud storage rsync -r -P "$out" "$dest/files" >> "$Q/sync.log" 2>&1 &&
       gcloud storage rsync -r -P "$Q" "$dest/queue" >> "$Q/sync.log" 2>&1; then
        log "synced to $dest"
    else
        log "sync to $dest FAILED (see sync.log)"
    fi
}

# 0 = stay up: a hold written this boot that has not expired, or (no such
# hold) still inside the boot grace period.
stay_up() {
    local btime
    btime=$(awk '/^btime/ {print $2}' /proc/stat)
    if [ -f "$Q/hold" ] && [ "$(stat -c %Y "$Q/hold")" -ge "$btime" ]; then
        [ "$(date +%s)" -lt "$(cat "$Q/hold")" ]
        return
    fi
    [ "$(cut -d. -f1 /proc/uptime)" -lt "$GRACE" ]
}

# First line with no done/failed state: prints "N<TAB>command".
next_job() {
    awk -v q="$Q" 'NF && $1 !~ /^#/ {
        if (system("test -e " q "/state/" NR ".done -o -e " q "/state/" NR ".failed")) {
            print NR "\t" $0; exit
        }
    }' "$Q/jobs.txt"
}

case ${1:-} in
setup)
    mkdir -p "$Q/state" "$Q/logs"
    touch "$Q/jobs.txt"
    echo $(( $(date +%s) + UP_HOLD )) > "$Q/hold"
    sudo tee /etc/systemd/system/bioembed-queue.service > /dev/null <<EOF
[Unit]
Description=BioEmbed job queue (scripts/gcp_runner.sh)
After=network-online.target
Wants=network-online.target

[Service]
User=$(id -un)
ExecStart=/bin/bash $HOME/BioEmbed/scripts/gcp_runner.sh run
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable --now bioembed-queue
    # The local eval env pins CPU torch; here use PyPI's CUDA build (as gcp_haystack.sh).
    curl -LsSf https://astral.sh/uv/install.sh | sh
    cd "$HOME/BioEmbed/eval"
    sed -i '/^\[tool.uv.sources\]/,/^explicit/d' pyproject.toml
    if ! { "$UV" sync --python 3.12 && "$UV" run python -c \
            'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'; }; then
        log "setup FAILED: no CUDA env; powering off"
        echo 0 > "$Q/hold"
        exit 1
    fi
    touch "$Q/ready"
    if [ -n "$(next_job)" ]; then echo 0 > "$Q/hold"; fi  # jobs came with `up`: no hold needed
    log "setup done"
    ;;
run)
    export PATH=$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
    cd "$HOME/BioEmbed/eval"
    log "runner started"
    while true; do
        job=$(next_job)
        if [ -n "$job" ] && [ -e "$Q/ready" ]; then
            n=${job%%$'\t'*} cmd=${job#*$'\t'}
            tries=$(( $(cat "$Q/state/$n.tries" 2>/dev/null || echo 0) + 1 ))
            echo "$tries" > "$Q/state/$n.tries"
            if [ "$tries" -gt "$TRIES" ]; then
                echo "not finished after $TRIES starts (VM restarts mid-job?)" > "$Q/state/$n.failed"
                log "job $n given up"
                continue
            fi
            log "job $n start (try $tries): $cmd"
            echo "$n" > "$Q/current"
            echo "=== $(date -u +%FT%TZ) try $tries: $cmd" >> "$Q/logs/$n.log"
            bash -o pipefail -c "$cmd" >> "$Q/logs/$n.log" 2>&1
            rc=$?
            rm -f "$Q/current"
            if [ "$rc" -eq 0 ]; then
                touch "$Q/state/$n.done"
            else
                { echo "exit $rc"; tr '\r' '\n' < "$Q/logs/$n.log" | grep -iv warn | tail -n 30; } > "$Q/state/$n.failed"
            fi
            log "job $n exit $rc"
            sync
            continue
        fi
        if ! stay_up; then
            log "nothing to run, powering off"
            sync
            sudo poweroff
            exit 0
        fi
        sleep 30
    done
    ;;
jobs|status)
    awk -v q="$Q" 'NF && $1 !~ /^#/ {
        s = "pending"
        if (!system("test -e " q "/state/" NR ".done")) s = "done"
        else if (!system("test -e " q "/state/" NR ".failed")) s = "FAILED"
        else if (!system("grep -qx " NR " " q "/current 2>/dev/null")) s = "running"
        printf "%3d %-8s %s\n", NR, s, substr($0, 1, 150)
    }' "$Q/jobs.txt"
    for f in "$Q"/state/*.failed; do [ -e "$f" ] && { echo "--- $(basename "$f")"; tail -n 5 "$f"; }; done
    if [ -e "$Q/current" ]; then
        echo "--- job $(cat "$Q/current") log tail"
        tr '\r' '\n' < "$Q/logs/$(cat "$Q/current").log" | grep -iv warn | grep . | tail -n 6
    fi
    [ -e "$Q/ready" ] || echo "(setup not finished: no ~/queue/ready)"
    [ "$1" = status ] || exit 0
    echo "--- service $(systemctl is-active bioembed-queue); $(uptime -p)"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    df -h --output=used,avail "$HOME" | tail -n 1
    ;;
*)
    echo "usage: $0 setup|run|status|jobs" >&2
    exit 2
    ;;
esac
