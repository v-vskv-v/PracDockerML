* Реализация веб-сервера, предоставляющего интерфейс взаимодействия с двумя реализованными композиционными моделями:

  - Gradient Boosting Decision Trees (MSE)
  - Random Forest (MSE)

* Веб-сервер реализован с помощью фреймворка Flask, а само решение обернуто в Docker контейнер.

* Формат загружаемых файлов:
    1. CSV-формат
    2. Только вещественные признаки (при наличии категориальных признаков в данных модели смогут с ними работать только как с вещественными числами)
    3. Де-факто матрица M x (N + 1), где первые N столбцов образуют матрицу объектов-признаков, а последний столбец - сопоставленные объектам в заимное соответствие вещественные числа

* Инструкция:
    1. Чтобы собрать докер образ: `docker build -t repo_name/image_name:image_tag .`
    2. Чтобы его запустить: `docker run --rm -p 5000:5000 -v "$PWD/WebServerML/data:/root/WebServerML/data" -v "$PWD/WebServerML/static:/root/WebServerML/static" -i repo_name/image_name`

* Docker-образ: https://hub.docker.com/r/vacbansry/webserver_gbdt_rf_regression
