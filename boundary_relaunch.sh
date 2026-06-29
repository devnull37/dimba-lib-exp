#!/usr/bin/env bash
# boundary_relaunch.sh — safe 3a→3b cutover orchestrator
#
# Usage:
#   boundary_relaunch.sh verify    read-only checks: checkpoint exists/stable/valid
#   boundary_relaunch.sh go-live   kill old trainer, launch 3b in tmux (calls verify first)
#
# NEVER run go-live unless verify passes.
# This script is idempotent under verify (pure reads, no side effects).
# go-live is irreversible — it kills the running trainer and starts a new one.

# ── config ────────────────────────────────────────────────────────────────────
CKPT="checkpoints/distill/distill_stage3a.pt"
REPO="devnull37/d1-135m-50b"
TMUX_SESSION="ssh_tmux"
TRAIN_PID_FILE="$HOME/dimba_train.pid"
LOG="$HOME/dimba_train.log"
REPO_DIR="/workspace/dimba-lib-exp"
VENV="/venv/main"

# Log noise patterns to suppress from training log tails
NOISE_FILTER='httpx|HTTP Request|cas-bridge'

# Minimum checkpoint size in bytes (100 MB)
MIN_CKPT_BYTES=$((100 * 1024 * 1024))

# ── helpers ───────────────────────────────────────────────────────────────────
red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
info()   { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die()    { red "FATAL: $*"; exit 1; }

# ── verify ────────────────────────────────────────────────────────────────────
cmd_verify() {
    local errors=0

    info "=== VERIFY: checking boundary checkpoint ==="

    # (a) Checkpoint exists and is large enough
    if [[ ! -f "$REPO_DIR/$CKPT" ]]; then
        red "  FAIL: checkpoint not found: $REPO_DIR/$CKPT"
        errors=$((errors + 1))
    else
        local size
        size=$(stat -c%s "$REPO_DIR/$CKPT" 2>/dev/null || stat -f%z "$REPO_DIR/$CKPT" 2>/dev/null)
        local mtime
        mtime=$(stat -c%y "$REPO_DIR/$CKPT" 2>/dev/null || stat -f "%Sm" "$REPO_DIR/$CKPT" 2>/dev/null)

        info "  checkpoint: $REPO_DIR/$CKPT"
        info "  size: ${size} bytes ($(( size / 1024 / 1024 )) MiB)"
        info "  mtime: ${mtime}"

        if (( size < MIN_CKPT_BYTES )); then
            red "  FAIL: checkpoint is only ${size} bytes — expected >100 MiB (still writing?)"
            errors=$((errors + 1))
        else
            green "  OK: checkpoint size ≥ 100 MiB"
        fi

        # (a) Stable size check — wait ~30s and compare
        info "  Checking stable size (sleeping 30s) …"
        local size1
        size1=$(stat -c%s "$REPO_DIR/$CKPT" 2>/dev/null)
        sleep 30
        local size2
        size2=$(stat -c%s "$REPO_DIR/$CKPT" 2>/dev/null)
        if [[ "$size1" != "$size2" ]]; then
            red "  FAIL: checkpoint size changed ($size1 → $size2) — still being written"
            errors=$((errors + 1))
        else
            green "  OK: checkpoint size is stable ($size2 bytes)"
        fi
    fi

    # (c) Grep training log for save/upload confirmation (filter noise)
    info "  Scanning training log for stage3a save events …"
    if [[ -f "$LOG" ]]; then
        local save_line
        save_line=$(grep -v -E "$NOISE_FILTER" "$LOG" 2>/dev/null | grep "distill_stage3a" | tail -5)
        if [[ -n "$save_line" ]]; then
            green "  Found stage3a save lines in log:"
            echo "$save_line" | while IFS= read -r line; do echo "    $line"; done
        else
            yellow "  WARNING: no 'distill_stage3a' save line found in $LOG (log may be elsewhere)"
        fi

        local upload_line
        upload_line=$(grep -v -E "$NOISE_FILTER" "$LOG" 2>/dev/null | grep -E "upload|huggingface.*distill_stage3a" | tail -5)
        if [[ -n "$upload_line" ]]; then
            green "  Found HF-upload lines:"
            echo "$upload_line" | while IFS= read -r line; do echo "    $line"; done
        else
            yellow "  WARNING: no HF upload confirmation found (upload may not be enabled)"
        fi
    else
        yellow "  WARNING: training log not found at $LOG — cannot scan log"
    fi

    # (d) CPU-only smoke test via generate.py
    if [[ -f "$REPO_DIR/$CKPT" ]]; then
        info "  Running CPU-only generate.py smoke test …"
        local gen_out
        # Use CUDA_VISIBLE_DEVICES="" to force CPU; the script auto-selects cpu when no CUDA
        gen_out=$(
            CUDA_VISIBLE_DEVICES="" python3 "$REPO_DIR/scripts/generate.py" \
                --checkpoint "$REPO_DIR/$CKPT" \
                --prompt "The quick brown fox jumps over" \
                --length 32 \
                --num-steps 20 \
                2>&1
        )
        local gen_exit=$?
        if (( gen_exit != 0 )); then
            red "  FAIL: generate.py exited with status $gen_exit"
            echo "$gen_out" | tail -20 | while IFS= read -r line; do echo "    $line"; done
            errors=$((errors + 1))
        else
            green "  OK: generate.py completed without error"
            echo "$gen_out" | while IFS= read -r line; do echo "    $line"; done
        fi
    fi

    # (e) Summary
    echo ""
    if (( errors == 0 )); then
        green "VERIFY OK — checkpoint is present, stable, and smoke-tested"
        return 0
    else
        red "VERIFY FAILED ($errors error(s)) — do NOT go-live until these are resolved"
        return 1
    fi
}

# ── go-live ───────────────────────────────────────────────────────────────────
cmd_go_live() {
    # Safety: enforce strict mode for the destructive phase
    set -euo pipefail

    info "=== GO-LIVE: starting 3a→3b boundary cutover ==="

    # First, run verify — abort on any failure
    cmd_verify || die "verify failed — aborting go-live"
    echo ""
    info "verify passed — proceeding with go-live"

    # (a) Read old PID
    local old_pid=""
    if [[ -f "$TRAIN_PID_FILE" ]]; then
        old_pid=$(cat "$TRAIN_PID_FILE")
        info "old trainer PID from $TRAIN_PID_FILE: $old_pid"
    else
        yellow "WARNING: $TRAIN_PID_FILE not found — will attempt to kill by name"
    fi

    # (b) Kill old trainer (TERM first, wait up to 30s, then KILL)
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
        info "Sending SIGTERM to pid $old_pid …"
        kill -TERM "$old_pid" || true
        local waited=0
        while kill -0 "$old_pid" 2>/dev/null && (( waited < 30 )); do
            sleep 1
            waited=$((waited + 1))
        done
        if kill -0 "$old_pid" 2>/dev/null; then
            yellow "Process $old_pid still alive after 30s — sending SIGKILL"
            kill -KILL "$old_pid" || true
            sleep 2
        fi
        if kill -0 "$old_pid" 2>/dev/null; then
            die "Could not kill pid $old_pid — aborting go-live"
        fi
        green "Process $old_pid terminated"
    else
        yellow "No live process at pid '$old_pid' — assuming already stopped"
    fi

    # (c) Confirm GPU freed: poll nvidia-smi for up to 30s
    info "Waiting for GPU to show the python process gone from nvidia-smi …"
    local waited=0
    while nvidia-smi | grep -q "python" && (( waited < 30 )); do
        sleep 2
        waited=$((waited + 2))
    done
    if nvidia-smi | grep -q "python"; then
        yellow "WARNING: a python process still shows in nvidia-smi — might be a different job"
    else
        green "GPU clear — no python process visible in nvidia-smi"
    fi
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true

    # (d) Launch 3b in tmux
    info "Launching 3b trainer in tmux session '$TMUX_SESSION', window 'train3b' …"
    local launch_cmd
    launch_cmd="source $VENV/bin/activate && cd $REPO_DIR && python3 scripts/train_h100.py --preset budget28_3b --phase all --checkpoint $CKPT --resume --hf-repo $REPO 2>&1 | tee -a $LOG"

    # Create a new window named train3b inside the existing tmux session
    tmux new-window -t "$TMUX_SESSION" -n "train3b" \; \
         send-keys -t "$TMUX_SESSION:train3b" "$launch_cmd" Enter

    info "tmux window 'train3b' created and launch command sent"

    # (e) Capture new PID — wait up to 60s for it to appear
    info "Waiting up to 60s for new python trainer to appear …"
    # Disable pipefail for the poll loop (pgrep may legitimately return nonzero)
    set +e
    local new_pid=""
    local waited=0
    while (( waited < 60 )); do
        sleep 3
        waited=$((waited + 3))
        new_pid=$(pgrep -f "train_h100.py.*budget28_3b" | head -1)
        if [[ -n "$new_pid" ]]; then
            break
        fi
    done
    set -e

    if [[ -z "$new_pid" ]]; then
        yellow "WARNING: could not find new trainer PID via pgrep — check tmux manually"
    else
        echo "$new_pid" > "$TRAIN_PID_FILE"
        green "New trainer PID: $new_pid — written to $TRAIN_PID_FILE"
    fi

    # (f) Sleep 60s then tail log for confirmation
    info "Sleeping 60s then checking log for 3b startup …"
    sleep 60

    echo ""
    info "--- Last 40 filtered log lines ---"
    # Use || true so pipefail doesn't fire on grep returning nonzero
    { grep -v -E "$NOISE_FILTER" "$LOG" 2>/dev/null | tail -40; } || true
    echo ""

    # Check for expected 3b markers
    local stage3_line
    stage3_line=$(grep -v -E "$NOISE_FILTER" "$LOG" 2>/dev/null | grep -E "Stage 3b|stage3.*UNFROZEN|lr=3e-05|budget28_3b|3b-only" | tail -5 || true)
    if [[ -n "$stage3_line" ]]; then
        green "3b phase confirmed in log:"
        echo "$stage3_line" | while IFS= read -r line; do echo "    $line"; done
    else
        yellow "WARNING: could not confirm 3b phase markers yet — check log manually:"
        yellow "  grep -v -E '$NOISE_FILTER' $LOG | tail -60"
    fi

    echo ""
    if [[ -n "$new_pid" ]]; then
        green "GO-LIVE OK / pid=$new_pid"
    else
        yellow "GO-LIVE LAUNCHED (PID unknown) — verify manually with: tmux a -t $TMUX_SESSION"
    fi
}

# ── dispatch ──────────────────────────────────────────────────────────────────
case "${1:-}" in
    verify)
        cd "$REPO_DIR"
        cmd_verify
        ;;
    go-live)
        cd "$REPO_DIR"
        cmd_go_live
        ;;
    *)
        echo "Usage: $0 {verify|go-live}"
        echo ""
        echo "  verify   — read-only checks: checkpoint present, stable, smoke-tested"
        echo "  go-live  — kill old trainer, launch 3b in tmux (calls verify first)"
        exit 1
        ;;
esac
