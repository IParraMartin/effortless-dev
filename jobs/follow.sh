#!/bin/bash
# `follow <jobid>` — tail a running Slurm job's output live.
#
# Add to ~/.bashrc:
#
#     source /global/scratch/users/iparra/effortless-dev/jobs/follow.sh
#
# Then:
#
#     follow 36350     # that job
#     follow           # your most recent job
#
# Paths come from `scontrol`, not from guessing the filename, so this works from
# any directory and for any job regardless of how its --output was written. The
# glob is only a fallback for jobs that have already left the queue, since
# scontrol forgets them a few minutes after they finish.

follow() {
    local jobid="${1:-}"

    if [ -z "$jobid" ]; then
        jobid="$(squeue -u "$USER" -h -o '%i' 2>/dev/null | tail -1)"
        if [ -z "$jobid" ]; then
            echo "follow: no jobs in the queue for $USER" >&2
            return 1
        fi
        echo "follow: newest job is $jobid"
    fi

    local info out err state name
    info="$(scontrol show job "$jobid" 2>/dev/null || true)"
    out="$(sed -n 's/.*StdOut=\([^[:space:]]*\).*/\1/p' <<<"$info" | head -1)"
    err="$(sed -n 's/.*StdErr=\([^[:space:]]*\).*/\1/p' <<<"$info" | head -1)"
    name="$(sed -n 's/.*JobName=\([^[:space:]]*\).*/\1/p' <<<"$info" | head -1)"

    # Job already gone from scontrol: fall back to the logs/%x_%j convention.
    #
    # `find` rather than a glob. An unmatched glob is left literal by bash but
    # aborts under zsh, and array subscripts differ between the two, so the
    # obvious `guess=(logs/*_$jobid.out); [ -e "${guess[0]}" ]` silently fails in
    # one shell and errors in the other.
    if [ -z "$out" ]; then
        out="$(find logs -maxdepth 1 -name "*_${jobid}.out" 2>/dev/null | head -1)"
        if [ -n "$out" ]; then
            err="${out%.out}.err"
            [ -e "$err" ] || err=""
        else
            echo "follow: job $jobid is not in the queue, and no" >&2
            echo "        logs/*_$jobid.out exists under $(pwd)." >&2
            echo "        Run it from the submit directory, or pass a live job id." >&2
            return 1
        fi
    fi

    # A pending job has no file yet. Say why rather than failing.
    if [ ! -e "$out" ]; then
        state="$(squeue -j "$jobid" -h -o '%T' 2>/dev/null)"
        echo "follow: job $jobid (${name:-?}) is ${state:-PENDING}; waiting for output..."
        while [ ! -e "$out" ]; do
            squeue -j "$jobid" -h -o '%T' >/dev/null 2>&1 || {
                echo "follow: job $jobid left the queue without writing $out" >&2
                return 1
            }
            sleep 5
        done
    fi

    local files=("$out")
    # Merged stdout/stderr is common; do not tail the same file twice.
    [ -n "$err" ] && [ "$err" != "$out" ] && files+=("$err")

    echo "follow: job $jobid (${name:-?})"
    printf 'follow:   %s\n' "${files[@]}"
    echo "follow: Ctrl-C to stop watching (the job keeps running)"
    echo

    # -F rather than -f: the file may be replaced or appear late, and -F keeps
    # up with that where -f silently follows a stale inode.
    tail -n 50 -F "${files[@]}" &
    local tail_pid=$!

    # Stop when the job leaves the queue, so this returns instead of hanging on
    # a finished job. Killed on Ctrl-C too, so no tail is left behind.
    trap 'kill "$tail_pid" 2>/dev/null; trap - INT TERM; return' INT TERM
    while squeue -j "$jobid" -h -o '%T' 2>/dev/null | grep -q .; do
        sleep 10
    done

    # Let the last writes land before cutting the stream.
    sleep 3
    kill "$tail_pid" 2>/dev/null
    wait "$tail_pid" 2>/dev/null
    trap - INT TERM

    echo
    echo "follow: job $jobid finished"
    sacct -j "$jobid" --format=JobID%14,JobName%18,State,Elapsed,MaxRSS 2>/dev/null | head -4
}
