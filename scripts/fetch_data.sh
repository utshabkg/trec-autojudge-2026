#!/usr/bin/env bash
# Fetch the AutoJudge datasets, reading the password from a file outside the repo.
#
# The release at trec-auto-judge.cs.unh.edu is behind HTTP basic auth as user
# `trec2026`, with the password of the TREC Active Participants page
# (https://trec.nist.gov/act_part/act_part.html, itself basic-auth protected).
# That is NOT your personal ir.nist.gov/evalbase account password, and evalbase
# now logs in through Login.gov so it cannot be scripted.
#
# TIRA is not an alternative route. `tira-cli download` serves only
# non-confidential datasets: kiddie returns 200, every pilot and both 2026 test
# sets return 405. Verified.
#
# Put the password in place once:
#   mkdir -p ~/.config/trec-autojudge && chmod 700 ~/.config/trec-autojudge
#   printf '%s' 'THE-PASSWORD' > ~/.config/trec-autojudge/trec_password
#   chmod 600 ~/.config/trec-autojudge/trec_password
#
# Then:  bash scripts/fetch_data.sh --pilot --test-2026
set -euo pipefail

PW_FILE="${TREC_PASSWORD_FILE:-$HOME/.config/trec-autojudge/trec_password}"

if [ ! -f "$PW_FILE" ]; then
    echo "No password file at $PW_FILE" >&2
    echo "See the header of this script for how to create it." >&2
    exit 1
fi

export TREC_AUTOJUDGE_USER="${TREC_AUTOJUDGE_USER:-trec2026}"
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
    echo "401 means the password is wrong. It is the shared TREC Active" >&2
    echo "Participants password, not your evalbase login." >&2
    exit 1
fi
echo "Auth OK. Fetching ${*:-all datasets} ..."

exec ./fetch_datasets.py "$@"
