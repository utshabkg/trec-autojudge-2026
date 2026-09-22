#!/usr/bin/env bash
# Fetch the AutoJudge datasets, reading the password from a file outside the repo.
#
# The release at trec-auto-judge.cs.unh.edu is behind HTTP basic auth. The user is
# `trec2026`, as the howto says.
#
# Read this if you hit a 401, because we lost time to it. Each TREC year has its
# OWN password, both sent round by Angela Ellis at NIST in that year's welcome
# mail. There is no single "Active Participants password". We first tried
# trec2026 with the 2025 password, got a 401, and wrongly concluded the server was
# misconfigured, because the 2025 pair happens to work on this UNH data host even
# though it is rejected by NIST. Measured on 2026-09-22:
#
#   trec2026 + 2026 password -> 200 on UNH, 200 on trec.nist.gov
#   trec2025 + 2025 password -> 200 on UNH, 401 on trec.nist.gov
#
# So a 401 means you have the wrong year's password, not a broken server. The
# password is the shared TREC Active Participants one for the year, NOT your
# personal ir.nist.gov/evalbase login, and evalbase now goes through Login.gov so
# it cannot be scripted anyway.
#
# TIRA is not an alternative route. `tira-cli download` serves only
# non-confidential datasets: kiddie returns 200, every pilot and both 2026 test
# sets return 405. Verified.
#
# Put the credentials in place once:
#   mkdir -p ~/.config/trec-autojudge && chmod 700 ~/.config/trec-autojudge
#   printf '%s' 'THE-PASSWORD' > ~/.config/trec-autojudge/trec_password
#   printf '%s' 'trec2026'     > ~/.config/trec-autojudge/trec_user
#   chmod 600 ~/.config/trec-autojudge/trec_{password,user}
#
# Then:  bash scripts/fetch_data.sh --pilot --test-2026
set -euo pipefail

CONF="${TREC_CONFIG_DIR:-$HOME/.config/trec-autojudge}"
PW_FILE="${TREC_PASSWORD_FILE:-$CONF/trec_password}"
USER_FILE="${TREC_USER_FILE:-$CONF/trec_user}"

if [ ! -f "$PW_FILE" ]; then
    echo "No password file at $PW_FILE" >&2
    echo "See the header of this script for how to create it." >&2
    exit 1
fi

if [ -z "${TREC_AUTOJUDGE_USER:-}" ]; then
    if [ -f "$USER_FILE" ]; then
        TREC_AUTOJUDGE_USER="$(cat "$USER_FILE")"
    else
        TREC_AUTOJUDGE_USER="trec2026"
    fi
fi
export TREC_AUTOJUDGE_USER
TREC_AUTOJUDGE_PASSWORD="$(cat "$PW_FILE")"
export TREC_AUTOJUDGE_PASSWORD

# Fail fast and legibly on a wrong password, rather than part way through a
# multi-gigabyte download.
probe() {
    curl -s -o /dev/null -w '%{http_code}' -I \
        -u "$TREC_AUTOJUDGE_USER:$TREC_AUTOJUDGE_PASSWORD" \
        "https://trec-auto-judge.cs.unh.edu/datareleases/v1.0.0/anonymized-runs-rag26-v1.tar.gz"
}
code="$(probe || true)"
if [ "$code" != "200" ]; then
    echo "Auth probe returned HTTP $code as user '$TREC_AUTOJUDGE_USER'." >&2
    echo "401 almost always means the wrong YEAR's password. Each TREC year has" >&2
    echo "its own, sent round by NIST in that year's welcome mail, and the user" >&2
    echo "and password must be from the same year. It is not an evalbase login." >&2
    echo "Check ~/.config/trec-autojudge/trec_user and trec_password agree." >&2
    exit 1
fi
echo "Auth OK. Fetching ${*:-all datasets} ..."

exec ./fetch_datasets.py "$@"
