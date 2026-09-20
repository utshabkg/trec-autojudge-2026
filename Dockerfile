# Corroborate: the TIRA code submission image for team IRIS-MST.
#
# The judge makes no LLM calls, so this image needs no network at runtime. What
# it does need is the encoder weights, and the TIRA sandbox reaches nothing but
# an allow-list of the organizers' own LLM endpoints. So the weights are baked in
# and the Hub is switched off, which also makes runs reproducible.
FROM ghcr.io/trec-auto-judge/trec-auto-judge-base:dev-0.0.1

# --------------------------------------------------------------------------
# Why this image does not use the base image's /venv
# --------------------------------------------------------------------------
# The base image's system Python, and therefore /venv, is 3.11.0rc1: a release
# candidate. torch's _dynamo polyfills do not survive on it, and any modern
# transformers reaches _dynamo during `from transformers import PreTrainedModel`:
#
#   transformers 5.x: modeling_utils -> integrations/deepgemm.py
#                     -> @torch._dynamo.assume_constant_result
#                     -> AttributeError: 'sys' has no attribute
#                        'get_int_max_str_digits'   (added in Python 3.11.1)
#
#   transformers 4.x: modeling_utils -> integrations/flex_attention.py
#                     -> @torch.compiler.disable -> import torch._dynamo
#                     -> ValueError: Duplicate dispatch rule for sys.intern
#
# Both were observed in CI. Capping transformers is therefore not a fix: the
# interpreter is the problem. So we build our own on a released Python and put it
# first on PATH, leaving the base image's /venv untouched.
RUN uv --version \
 && uv python install 3.12 \
 && uv venv --python 3.12 /venv312
ENV PATH="/venv312/bin:$PATH" \
    VIRTUAL_ENV="/venv312"

# README.md is not optional here: pyproject.toml declares `readme = "README.md"`,
# so an editable install fails without it.
ADD judges /auto-judge/judges
ADD pyproject.toml README.md /auto-judge/

WORKDIR /auto-judge

# CPU torch explicitly, and first: resolving `torch` from PyPI pulls a CUDA build
# with several GB of nvidia-* wheels that a TIRA worker cannot use, and doing it
# before the project install stops sentence-transformers dragging it back in.
RUN uv pip install --python /venv312/bin/python \
        torch --index-url https://download.pytorch.org/whl/cpu \
 && uv pip install --python /venv312/bin/python -e '.[all]'

# Prove the import chain at build time rather than discovering it half way
# through a run. This check is what turned the two failures above from confusing
# runtime errors into a build that stops with the reason on screen.
RUN python -c "\
import sys, torch, transformers, sentence_transformers as st; \
from transformers import PreTrainedModel, AutoModelForSequenceClassification; \
from sentence_transformers import SentenceTransformer; \
import torch._dynamo; \
print('import chain OK on Python', sys.version.split()[0], \
      '| torch', torch.__version__, \
      '| transformers', transformers.__version__, \
      '| sentence-transformers', st.__version__)" \
 && which auto-judge \
 && auto-judge --help > /dev/null \
 && echo "auto-judge resolves from /venv312"

# Bake the encoder weights. snapshot_download rather than loading through
# sentence-transformers, so the build does not depend on how a given
# sentence-transformers version chooses to construct a model.
ENV HF_HOME=/models
RUN python -c "\
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

# And prove the weights load with the Hub switched off, which is how they will be
# loaded in the sandbox. A missing file surfaces here, not mid-run.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
SentenceTransformer('BAAI/bge-base-en-v1.5', device='cpu', local_files_only=True); \
AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base', local_files_only=True); \
AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base', local_files_only=True); \
print('offline model load OK')"

# git metadata for provenance (tira's runtime stats look for a repo at ./)
ADD .git /auto-judge/.git
