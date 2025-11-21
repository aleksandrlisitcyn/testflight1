# Changelog

All notable changes to this project will be documented in this file.

## How to update this file

- Run `python tools/update_changelog.py`
- The script reads recent `git log` entries and rewrites the `[Unreleased]` section in-place.
  Older sections are preserved verbatim.

## [Unreleased]

### Added
- `pattern_mode` flag (color/symbol/mixed) threaded through the API, UI and full pipeline.
- Sanity report metadata that surfaces palette/grid anomalies without mutating results.
- Project documentation folder (`docs/overview.md`, `docs/pipeline.md`, `docs/api.md`) plus an
  automated changelog helper (`tools/update_changelog.py`).

### Changed
- Neutralised `detail_level` so recognition stays faithful; behaviour differences now come only from
  `pattern_mode`.
- Removed global colour collapsing/background removal – consolidation happens strictly per cell.
- Local sampling now uses robust trimmed statistics and preserves dominant hues in antique charts.

### Fixed
- PDF exporter gracefully falls back to built-in fonts if `assets/fonts/DejaVuSans.ttf` is invalid.

## [Legacy]

### Added
- Детектор ROI/сетки с оценкой уверенности и перспективной коррекцией.
- Устойчивое усреднение клеток, подавление фона, пост-объединение цветов и fallback на минимальную схему.
- Предпросмотр символов с акцентированными 10× линиями, PDF-легенда на DejaVuSans.
- Набор синтетических бенчей (`data/bench`) и скрипт `scripts/run_bench.py` с PNG-артефактами.
- Конфигурация `pre-commit`, `pyproject.toml`, `mypy.ini`, dev-зависимости и GitHub Actions pipeline.
- Набор unit/integration тестов (ROI, grid, sampling, pipeline, символы).

### Changed
- Обновлён фронтенд (`app/templates/index.html`) в стиле macOS utility.
- Pipeline использует новые эвристики detail_level, валидирует meta и хранит legend.
- PDF экспортер рендерит легенду с символами и одинаковым шрифтом.
