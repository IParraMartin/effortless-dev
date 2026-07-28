#!/bin/bash
# `check` — what every one of your Slurm jobs is actually doing, right now.
#
# Add to ~/.bashrc:
#
#     source /global/scratch/users/iparra/effortless-dev/jobs/check.sh
#
# Then:
#
#     check              # every job you have queued or running
#     check 35926835     # one job
#     check vr-probe1    # every job whose name matches
#
# Every path is resolved **per job**, from `scontrol show job`, and never from a
# fixed location. That is the whole design constraint: a monitor that reads
# progress out of one hardcoded directory reports whatever last wrote there,
# attributed to whichever job it happened to be looking at — so a finished
# experiment from another project shows up as live progress under the name of a
# job that has barely started.
#
# It also means this works for any job, in any repository, with no per-project
# configuration.

# Seconds since a file was last written. `stat` takes different flags on GNU and
# BSD, and this is sourced into interactive shells on both.
_check_age() {
    local file="$1" mtime now
    mtime="$(stat -c %Y "$file" 2>/dev/null || stat -f %m "$file" 2>/dev/null)"
    [ -n "$mtime" ] || return 1
    now="$(date +%s)"
    echo $(( now - mtime ))
}

# "3m", "2h10m" — a duration worth reading at a glance.
_check_duration() {
    local s="$1"
    if [ "$s" -lt 60 ]; then echo "${s}s"
    elif [ "$s" -lt 3600 ]; then echo "$(( s / 60 ))m"
    else echo "$(( s / 3600 ))h$(( (s % 3600) / 60 ))m"
    fi
}

_check_ago() { echo "$(_check_duration "$1") ago"; }

_check_one() {
    local jobid="$1" name="$2" state="$3" elapsed="$4" limit="$5" where="$6"
    local info out err age line steps

    printf '%-11s %-20s %-3s %s/%s  %s\n' \
        "$jobid" "$name" "${state:0:3}" "$elapsed" "$limit" "$where"

    # Pending jobs have no output yet; the reason is already in `where`.
    if [ "$state" != "RUNNING" ] && [ "$state" != "COMPLETING" ]; then
        return
    fi

    info="$(scontrol show job "$jobid" 2>/dev/null || true)"
    out="$(sed -n 's/.*StdOut=\([^[:space:]]*\).*/\1/p' <<<"$info" | head -1)"
    err="$(sed -n 's/.*StdErr=\([^[:space:]]*\).*/\1/p' <<<"$info" | head -1)"

    if [ -z "$out" ] || [ ! -e "$out" ]; then
        echo "     log      (no output file yet)"
        return
    fi

    age="$(_check_age "$out")"
    printf '     log      %s   %s\n' "$out" "$(_check_ago "${age:-0}")"

    # A running job whose log has gone quiet is the failure this is for: it
    # looks identical to a healthy one in squeue.
    if [ -n "$age" ] && [ "$age" -gt 600 ]; then
        echo "     !!       no output for $(_check_duration "$age") while RUNNING — possibly stalled"
    fi

    # Last thing the job said. Generic on purpose: any job, any format.
    line="$(grep -v '^[[:space:]]*$' "$out" 2>/dev/null | tail -1)"
    [ -n "$line" ] && printf '     latest   %s\n' "${line:0:100}"

    # If it looks like a training loop, say how fast it is going. Derived from
    # the log's own step lines rather than assumed.
    steps="$(grep -oE '^step +[0-9]+' "$out" 2>/dev/null | tail -1 | grep -oE '[0-9]+')"
    if [ -n "$steps" ] && [ "$steps" -gt 0 ]; then
        local secs
        secs="$(_check_elapsed_seconds "$elapsed")"
        if [ -n "$secs" ] && [ "$secs" -gt 0 ]; then
            printf '     rate     step %s after %s  ->  %s steps/min\n' \
                "$steps" "$elapsed" "$(( steps * 60 / secs ))"
        fi
    fi

    # Anything that looks like a failure, in either stream. Cheap and it is the
    # thing you most want surfaced without reading the whole log.
    local problems=0 count file
    for file in "$out" ${err:+"$err"}; do
        [ -e "$file" ] || continue
        count="$(grep -cE 'Traceback|CUDA out of memory|Error|FAILED|Killed' \
            "$file" 2>/dev/null || true)"
        problems=$(( problems + ${count:-0} ))
    done
    if [ "$problems" -gt 0 ]; then
        echo "     !!       $problems line(s) matching Traceback/OOM/Error — check the log"
    fi
}

# Slurm elapsed times come as [DD-]HH:MM:SS or MM:SS.
_check_elapsed_seconds() {
    local t="$1" days=0
    case "$t" in
        *-*) days="${t%%-*}"; t="${t#*-}" ;;
    esac
    # Walk the colon-separated fields with parameter expansion rather than
    # `read -a`, which is spelled `-a` in bash and `-A` in zsh and so fails in
    # whichever one you did not write it for. 10# forces base ten: "08" is not
    # a valid octal literal and would abort the arithmetic.
    local total=0 part
    while [ -n "$t" ]; do
        part="${t%%:*}"
        total=$(( total * 60 + 10#${part:-0} ))
        [ "$t" = "$part" ] && break
        t="${t#*:}"
    done
    echo $(( total + days * 86400 ))
}

# Drop any same-named alias before defining the function.
#
# Bash expands aliases *while parsing*, so if `check` is already an alias the
# line `check() {` is rewritten before the parser reaches the parenthesis and
# fails with "syntax error near unexpected token `('". The message points at
# this file, which is misleading: nothing here is wrong, the name was simply
# taken. Interactive shells commonly have such an alias from another project.
unalias check 2>/dev/null || true

check() {
    local filter="${1:-}"
    local rows

    rows="$(squeue -u "$USER" -h -o '%i|%j|%T|%M|%l|%R' 2>/dev/null)"
    if [ -z "$rows" ]; then
        echo "check: no jobs queued or running for $USER"
        return 0
    fi

    local shown=0
    while IFS='|' read -r jobid name state elapsed limit where; do
        [ -n "$jobid" ] || continue
        if [ -n "$filter" ] \
           && [ "$jobid" != "$filter" ] \
           && ! printf '%s' "$name" | grep -q -- "$filter"; then
            continue
        fi
        if [ "$shown" -eq 0 ]; then
            printf '%-11s %-20s %-3s %s  %s\n' \
                "JOBID" "NAME" "ST" "ELAPSED/LIMIT" "NODE / REASON"
            printf '%s\n' \
                "------------------------------------------------------------------------"
        fi
        _check_one "$jobid" "$name" "$state" "$elapsed" "$limit" "$where"
        shown=$(( shown + 1 ))
    done <<<"$rows"

    if [ "$shown" -eq 0 ]; then
        echo "check: nothing matching '$filter'"
    fi
}
