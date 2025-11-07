# Stitch Converter Service

Сервис для автоматической оцифровки схем вышивки крестиком.
На вход — JPG/PNG (300 dpi+). На выход — .saga, .xsd/.xsp/.css/.dize, .pdf, .json, .csv.

## Стек
Python 3.10+, FastAPI, OpenCV, Pillow, scikit-learn, Jinja2, WeasyPrint.

## Запуск
```bash
docker-compose up --build
```

## API (основное)
- `POST /api/v1/jobs` — загрузка схемы, запуск обработки
- `GET /api/v1/jobs/{id}` — статус
- `GET /api/v1/jobs/{id}/preview?mode=color|symbols` — превью схемы
- `GET /api/v1/jobs/{id}/legend` — легенда
- `PATCH /api/v1/jobs/{id}/meta` — правка названия/описания
- `PATCH /api/v1/jobs/{id}/legend` — правка легенды
- `POST /api/v1/jobs/{id}/export` — экспорт (.saga/.pdf/.json/.csv/и др.)
