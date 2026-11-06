#!/usr/bin/env bash
# Reproduces the Jenkinsfile CI stages on a developer machine without Jenkins.
# Docker stages run only when a Docker daemon is available.
set -euo pipefail
cd "$(dirname "$0")/.."

stage() { printf '\n==================== [%s] ====================\n' "$1"; }

stage "Set up Python"
python3 -m venv .venv
. .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements-dev.txt
pip install --quiet -e .

stage "Lint (black + flake8)"
black --check kaz_fin_nlp tests
flake8 kaz_fin_nlp tests --max-line-length=88 --extend-ignore=E203,W503 && echo "flake8: 0 issues"

stage "Type check (mypy)"
mypy kaz_fin_nlp

stage "Security (bandit)"
bandit -r kaz_fin_nlp -ll -q && echo "bandit: no medium/high issues"

stage "Unit tests"
mkdir -p reports
pytest -q -o addopts="" --junitxml=reports/junit.xml --cov=kaz_fin_nlp \
       --cov-report=xml:reports/coverage.xml --cov-report=term-missing \
       --cov-fail-under=90 -p no:warnings

stage "Benchmark (NFR1 > 1000 articles/min)"
kaz-fin-nlp benchmark --iterations 2000

if docker info >/dev/null 2>&1; then
  stage "Build Docker image"
  docker build -t kaz-fin-nlp-core:local .
  stage "Smoke test container"
  docker rm -f kfn-smoke 2>/dev/null || true
  docker run -d --name kfn-smoke -p 8000:8000 kaz-fin-nlp-core:local
  sleep 4
  curl -fsS http://localhost:8000/health && echo
  docker rm -f kfn-smoke
else
  stage "Docker stages skipped (no Docker daemon)"
fi

stage "PIPELINE SUCCESS"
