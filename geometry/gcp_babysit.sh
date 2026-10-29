#!/usr/bin/env bash
# Keep geometry/umap_sweep.py going on a Spot VM: restart the VM after a preemption (retrying while the
# zone has no capacity), relaunch the sweep (it resumes from results.csv), and print one line per
# finished setting. Exits when the sweep prints "exit". Usage: geometry/gcp_babysit.sh VM ZONE [SEEN]
set -u
VM=$1 ZONE=$2 seen=${3:-0}  # (3rd arg: lines already reported, so a re-armed watch skips them)
ssh() { timeout 90 gcloud compute ssh "$VM" --zone "$ZONE" --command "$1" 2>/dev/null; }
while true; do
    st=$(gcloud compute instances describe "$VM" --zone "$ZONE" --format='value(status)' 2>/dev/null)
    if [ "$st" = TERMINATED ] || [ "$st" = STOPPED ]; then
        if gcloud compute instances start "$VM" --zone "$ZONE" >/dev/null 2>&1; then
            echo "$(date -u +%H:%MZ) $VM restarted"
            until ssh true; do sleep 10; done
        else
            sleep 120; continue
        fi
    fi
    out=$(ssh 'grep -E "^\{|^exit|^failed|Traceback|Error|Killed" ~/sweep.log')
    n=$(printf "%s" "$out" | grep -c .)
    if [ "$n" -gt "$seen" ]; then printf "%s\n" "$out" | tail -n +$((seen + 1)) | cut -c1-260; seen=$n; fi
    printf "%s" "$out" | grep -q "^exit" && break
    if ! ssh 'pgrep -f "[u]map_sweep.py" >/dev/null'; then
        echo "$(date -u +%H:%MZ) relaunching sweep"
        ssh 'cd ~/geometry && (setsid nohup sh -c "~/.local/bin/uv run python umap_sweep.py --data ~/medembed_small --grid all; echo exit \$?" < /dev/null >> ~/sweep.log 2>&1 &)' || true
    fi
    sleep 120
done
