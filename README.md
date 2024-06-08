# Стажировка в "Газпром-Медиа Холдинг"

## Тестовое задание: 
* Реализовать ML-модель классификации изображений с помощью PyTorch
*  Доп. задание: реализовать сервис на Flask/FastAPI, который может классифицировать блюда по картинке

## Навигация по репозиторию
* ml - папка с jupyter notebook, обработкой данных, тренировкой и валидацией модели
* front_and_back - папка, которая содержит файлы для web-сервиса, который написан на Flask

## Выполненные задачи:
* Обучена модель с точностью accuracy 0.87
* Реализован web-сервис на Flask

### локальный запуск: 
* `git clone https://github.com/CyberGeo335/gazprom-media-holding.git`
* перехоим по `cd .\gazprom-media-holding\front_and_back\`
* пишем `docker-compose build`, а затем `docker-compose up`

## Дополнительный функционал:
* Был сделан deploy на сервере selectel: `http://45.89.189.247:5000/`, активен до 18.06.2024
