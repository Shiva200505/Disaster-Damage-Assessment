# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-04-22

### Added
- Complete modern React SPA built with Vite + TypeScript using state-of-the-art UI practices and Framer Motion caching
- Standalone Evaluator suite (`member5_evaluation/evaluate_full.py`) allowing generalized full tile evaluations alongside static plots 
- Centralized tracking pipeline utilizing native `logging` pushing structured JSON payload files to storage whilst mirroring `Weights and Biases` natively inside the evaluation epoch callbacks
- Global Pytest configuration scaling functional verifications across architectures (`SiameseUNetV2/V3`), pipeline inference endpoints, and mathematical matrices boundaries
- Production robust Pydantic structured environment mappings directly isolating infrastructure limits locally in `.env`
- Graceful shutdown Celery event hooks terminating orphaned tasks preemptively during worker revocation interrupts
- GPU-Native Dummy-Pass Model Warmup procedure bound strictly alongside FastAPI's main execution lifecycles scaling cold-start performance natively
- Pre-save File header verification logic mapping EXIF scrubbing routines implicitly avoiding malicious file embedding before caching payloads locally
- Comprehensive Nginx Security Headers guaranteeing restricted origin frame scoping (`X-Frame-Options`, `CSP`, `Strict-Transport-Security`)
- GitHub Actions CI/CD workflows combining `ruff`, `mypy`, testing coverage tracing matrices natively into `codecov` endpoints.

### Changed
- Shifted away from legacy `Flask` and `Jinja2` bindings unifying infrastructure cleanly atop `FastAPI` + `React` deployments natively tracked through `docker-compose` frameworks 

### Fixed
- Stabilized metric evaluations enforcing native epsilon boundaries (`1e-7`) during Intersection over Union matrix resolutions
