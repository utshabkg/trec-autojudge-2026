# Corroborate: the TIRA code submission image for team iris.
#
# The judge makes no LLM calls, so this image needs no network at runtime. What
# it does need is the encoder weights, and the TIRA sandbox reaches nothing but
# an allow-list of the organizers' own LLM endpoints. So the weights are baked in
# and the Hub is switched off, which also makes runs reproducible.
FROM ghcr.io/trec-auto-judge/trec-auto-judge-base:dev-0.0.1

# README.md is not optional here: pyproject.toml declares `readme = "README.md"`,
# so an editable install fails without it.
ADD judges /auto-judge/judges
ADD pyproject.toml README.md /auto-judge/

WORKDIR /auto-judge

# Install into the base image's /venv (its PATH runs /venv/bin). A --system
# install would be invisible at runtime, which is the top troubleshooting item
# in the submission HowTo.
#
# CPU torch first, and explicitly: resolving `torch` from PyPI pulls a CUDA
# build with several GB of nvidia-* wheels that a TIRA worker cannot use. Doing
# it before the project install means sentence-transformers finds torch already
# satisfied and does not drag the CUDA variant back in.
RUN . /venv/bin/activate \
 && uv pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cpu \
 && uv pip install -e .[all]

# Bake the encoder weights. snapshot_download rather than loading through
# sentence-transformers, so the build does not depend on how a given
# sentence-transformers version chooses to construct a model.
ENV HF_HOME=/models
RUN . /venv/bin/activate && python -c "\
from huggingface_hub import snapshot_download; \
[snapshot_download(r) for r in ( \
    'BAAI/bge-base-en-v1.5', \
    'cross-encoder/nli-deberta-v3-base', \
    'BAAI/bge-reranker-v2-m3', \
)]" \
 && du -sh /models

# From here the Hub is unreachable and must never be needed. If a model is
# missing, the run should fail loudly at load time rather than hang on a network
# call that the sandbox will refuse anyway.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

# git metadata for provenance (tira's runtime stats look for a repo at ./)
ADD .git /auto-judge/.git
