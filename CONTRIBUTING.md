# Contributing to Kazakhstani NLP Financial News System

Thank you for contributing to the **Kazakhstani NLP Financial News System (Core Engine)**!

This project adheres to scientific software engineering standards, continuous integration practices, and test-driven development (TDD).

---

## 1. Code of Conduct

Please maintain professional, respectful, and constructive communication at all times.

---

## 2. Git Branching Strategy

We follow the **Git Flow** workflow:

- `main`: Production-ready releases. Protected branch.
- `develop`: Integration branch for tested features.
- `feature/<name>`: New capabilities or computational improvements.
- `fix/<name>`: Bug fixes and performance patches.

### Branch Workflow
1. Branch off `develop`: `git checkout -b feature/improved-kazakh-stemmer develop`
2. Implement features and write comprehensive unit tests.
3. Validate locally: ensure tests pass with **>90% coverage** and linters report 0 errors.
4. Push and submit a Pull Request targeting `develop`.

---

## 3. Commit Message Conventions

We strictly follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New computational feature or component.
- `fix:` Bug fix or regex correction.
- `docs:` Documentation updates.
- `test:` Adding or updating unit/integration tests.
- `refactor:` Code refactoring without functional alterations.
- `perf:` Throughput or latency optimization.
- `ci:` Changes to GitHub Actions workflows.

---

## 4. Development & Quality Standards

Before committing, ensure your changes pass all local verification steps:

```bash
# 1. Format code with Black
black kaz_fin_nlp tests

# 2. Check style with Flake8
flake8 kaz_fin_nlp tests --max-line-length=88 --extend-ignore=E203,W503

# 3. Static type check with Mypy
mypy kaz_fin_nlp

# 4. Security SAST audit with Bandit
bandit -r kaz_fin_nlp

# 5. Run test suite with coverage enforcement
pytest --cov=kaz_fin_nlp --cov-fail-under=90
```

All Pull Requests require automated CI validation across Python 3.10, 3.11, and 3.12 prior to merge.
