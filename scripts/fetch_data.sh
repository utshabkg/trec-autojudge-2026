#!/usr/bin/env bash
# Fetch the AutoJudge datasets, reading the password from a file outside the repo.
#
# The release at trec-auto-judge.cs.unh.edu is behind HTTP basic auth. The howto
# documents the user as `trec2026`, but that returns 401: the working user is
# `trec2025`, verified against both the index and a tarball. So the user is read
# from a file too rather than hardcoded to the documented value.
#
# The password is the one for the TREC Active Participants page
# (https://trec.nist.gov/act_part/act_part.html, itself basic-auth protected).
# That is NOT your personal ir.nist.gov/evalbase account password, and evalbase
# now logs in through Login.gov so it cannot be scripted.
#
# TIRA is not an alternative route. `tira-cli download` serves only
# non-confidential datasets: kiddie returns 200, every pilot and both 2026 test
# sets return 405. Verified.
#
# Put the credentials in place once:
#   mkdir -p ~/.config/trec-autojudge && chmod 700 ~/.config/trec-autojudge
#   printf '%s' 'THE-PASSWORD' > ~/.config/trec-autojudge/trec_password
#   printf '%s' 'trec2025'     > ~/.config/trec-autojudge/trec_user
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
        TREC_AUTOJUDGE_USER="trec2025"
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
    echo "401 means the user or password is wrong. The password is the shared" >&2
    echo "TREC Active Participants one, not an evalbase login, and the working" >&2
    echo "user is trec2025 even though the howto documents trec2026." >&2
    exit 1
fi
echo "Auth OK. Fetching ${*:-all datasets} ..."

exec ./fetch_datasets.py "$@"
