# Changelog

## Unreleased
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
