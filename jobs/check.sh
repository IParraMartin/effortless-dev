#!/bin/bash
# `check` — what every one of your Slurm jobs is doing, one line each.
#
# Add to ~/.bashrc (after any same-named alias):
#
#     source /global/scratch/users/iparra/effortless-dev/jobs/check.sh
#
# Then:
#
#     check              # every job you have queued or running
#     check 35936978     # one job
#     check vr-          # every job whose name matches
#     check -v           # add log paths and node names
#
# Every log path is resolved **per job**, from `scontrol show job`, never from a
# fixed location. A monitor pointed at one hardcoded directory reports whatever
# last wrote there under whichever job it is displaying, so a finished run from
# another project appears as live progress on a job that has barely started.
# Deriving the path from Slurm makes that unrepresentable and makes this work
# for any job in any repository, with no per-project configuration.

# Drop any same-named alias before defining the function. Bash expands aliases
# *while parsing*, so if `check` is already an alias the definition line is
# rewritten before the parser reaches the parenthesis and fails with "syntax
# error near unexpected token `('" — pointing at this file, though nothing here
# is wrong. An interactive shell commonly has such an alias from another project.
unalias check 2>/dev/null || true

# Slurm elapsed and limit strings, as [DD-]HH:MM:SS or MM:SS, into seconds.
# Parsed with parameter expansion rather than `read -a`, which is spelled `-a`
# in bash and `-A` in zsh and so breaks in whichever one it was not written for.
_check_seconds() {
    local t="$1" days=0 total=0 part
    case "$t" in
        *INFINITE*|*UNLIMITED*|"") echo ""; return ;;
        *-*) days="${t%%-*}"; t="${t#*-}" ;;
    esac
    while [ -n "$t" ]; do
        part="${t%%:*}"
        # 10# forces base ten: "08" is not a valid octal literal.
        total=$(( total * 60 + 10#${part:-0} ))
        [ "$t" = "$part" ] && break
        t="${t#*:}"
    done
    echo $(( total + days * 86400 ))
}

_check_duration() {
    local s="$1"
    if [ "$s" -lt 60 ]; then echo "${s}s"
    elif [ "$s" -lt 3600 ]; then echo "$(( s / 60 ))m"
    else echo "$(( s / 3600 ))h$(( (s % 3600) / 60 ))m"
    fi
}

_check_age() {
    local mtime
    mtime="$(stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null)"
    [ -n "$mtime" ] && echo $(( $(date +%s) - mtime ))
}

# One compact progress string from a training log.
#
# Rate comes from the *difference* between the last two step lines, never from
# total steps over job elapsed: the latter charges queue-to-start, model
# construction and the first data load against the training rate, which
# understated a real run by 55% — enough to make an 18-hour job look like 27.
_check_progress() {
    local out="$1" target
    target="$(grep -oE 'over [0-9]+ steps' "$out" 2>/dev/null | tail -1 \
        | grep -oE '[0-9]+')"

    grep -E '^step +[0-9]+' "$out" 2>/dev/null | tail -2 | awk -v target="${target:-0}" '
        {
            step[NR] = $2
            for (i = 1; i <= NF; i++) {
                if ($i == "loss") loss = $(i + 1)
                if ($i ~ /^[0-9.]+s$/) secs[NR] = substr($i, 1, length($i) - 1)
            }
        }
        END {
            if (NR == 0) exit
            printf "step %s", step[NR]
            if (loss != "") printf "  loss %s", loss
            if (NR < 2 || secs[2] <= secs[1]) { printf "\n"; exit }
            rate = (step[2] - step[1]) / (secs[2] - secs[1])
            if (rate <= 0) { printf "\n"; exit }
            printf "  %.0f/min", rate * 60
            if (target > step[2]) printf "  eta %.1fh", (target - step[2]) / rate / 3600
            printf "\n"
        }'
}

# Seconds of work still ahead, for comparing against the wall clock.
_check_eta_seconds() {
    local out="$1" target
    target="$(grep -oE 'over [0-9]+ steps' "$out" 2>/dev/null | tail -1 \
        | grep -oE '[0-9]+')"
    [ -n "$target" ] || return

    grep -E '^step +[0-9]+' "$out" 2>/dev/null | tail -2 | awk -v target="$target" '
        {
            step[NR] = $2
            for (i = 1; i <= NF; i++)
                if ($i ~ /^[0-9.]+s$/) secs[NR] = substr($i, 1, length($i) - 1)
        }
        END {
            if (NR < 2 || secs[2] <= secs[1]) exit
            rate = (step[2] - step[1]) / (secs[2] - secs[1])
            if (rate > 0 && target > step[2]) printf "%d", (target - step[2]) / rate
        }'
}

_check_one() {
    local jobid="$1" name="$2" state="$3" elapsed="$4" limit="$5" where="$6"
    local verbose="$7"
    local short detail out err age eta remaining problems=0 count file

    case "$state" in
        RUNNING)    short=RUN ;;
        PENDING)    short=PEND ;;
        COMPLETING) short=COMP ;;
        SUSPENDED)  short=SUSP ;;
        *)          short="${state:0:4}" ;;
    esac

    if [ "$state" != "RUNNING" ] && [ "$state" != "COMPLETING" ]; then
        printf '%-9s %-15s %-4s %8s  %s\n' "$jobid" "$name" "$short" "-" "$where"
        return
    fi

    out="$(scontrol show job "$jobid" 2>/dev/null \
        | sed -n 's/.*StdOut=\([^[:space:]]*\).*/\1/p' | head -1)"
    err="$(scontrol show job "$jobid" 2>/dev/null \
        | sed -n 's/.*StdErr=\([^[:space:]]*\).*/\1/p' | head -1)"

    detail=""
    if [ -n "$out" ] && [ -e "$out" ]; then
        detail="$(_check_progress "$out")"
        # No step lines yet: fall back to whatever the job last said, trimmed
        # of leading space so continuation lines do not look like columns.
        if [ -z "$detail" ]; then
            detail="$(grep -v '^[[:space:]]*$' "$out" | tail -1 \
                | sed 's/^[[:space:]]*//' | cut -c1-58)"
        fi
    else
        detail="(no output yet)"
    fi

    printf '%-9s %-15s %-4s %8s  %s\n' \
        "$jobid" "$name" "$short" "$elapsed" "$detail"

    # Warnings, indented, and only when they apply.
    if [ -n "$out" ] && [ -e "$out" ]; then
        age="$(_check_age "$out")"
        if [ -n "$age" ] && [ "$age" -gt 600 ]; then
            printf '%42s!! silent for %s — possibly stalled\n' "" \
                "$(_check_duration "$age")"
        fi

        # Work remaining against clock remaining. This is the warning worth
        # having: a job that will hit its wall clock before it finishes looks
        # completely healthy until the moment it is killed.
        eta="$(_check_eta_seconds "$out")"
        remaining="$(_check_seconds "$limit")"
        if [ -n "$eta" ] && [ -n "$remaining" ]; then
            remaining=$(( remaining - $(_check_seconds "$elapsed") ))
            if [ "$eta" -gt "$remaining" ]; then
                printf '%42s!! needs %s but only %s left on the clock\n' "" \
                    "$(_check_duration "$eta")" "$(_check_duration "$remaining")"
            fi
        fi

        for file in "$out" ${err:+"$err"}; do
            [ -e "$file" ] || continue
            count="$(grep -cE 'Traceback|CUDA out of memory|FAILED|Killed' \
                "$file" 2>/dev/null || true)"
            problems=$(( problems + ${count:-0} ))
        done
        [ "$problems" -gt 0 ] && \
            printf '%42s!! %s line(s) matching Traceback/OOM/Killed\n' "" "$problems"

        [ -n "$verbose" ] && printf '%42s%s\n' "" "$out"
    fi
}

check() {
    local filter="" verbose="" arg
    for arg in "$@"; do
        case "$arg" in
            -v|--verbose) verbose=1 ;;
            *) filter="$arg" ;;
        esac
    done

    local rows
    rows="$(squeue -u "$USER" -h -o '%i|%j|%T|%M|%l|%R' 2>/dev/null)"
    if [ -z "$rows" ]; then
        echo "check: no jobs queued or running for $USER"
        return 0
    fi

    local shown=0
    while IFS='|' read -r jobid name state elapsed limit where; do
        [ -n "$jobid" ] || continue
        if [ -n "$filter" ] && [ "$jobid" != "$filter" ] \
           && ! printf '%s' "$name" | grep -q -- "$filter"; then
            continue
        fi
        if [ "$shown" -eq 0 ]; then
            printf '%-9s %-15s %-4s %8s  %s\n' \
                "JOBID" "NAME" "ST" "ELAPSED" "PROGRESS / REASON"
        fi
        _check_one "$jobid" "$name" "$state" "$elapsed" "$limit" "$where" "$verbose"
        shown=$(( shown + 1 ))
    done <<<"$rows"

    [ "$shown" -eq 0 ] && echo "check: nothing matching '$filter'"
    return 0
}
