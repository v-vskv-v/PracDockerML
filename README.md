* Реализация веб-сервера, предоставляющего интерфейс взаимодействия с двумя реализованными композиционными моделями:

  - Gradient Boosting Decision Trees (MSE)
  - Random Forest (MSE)

* Веб-сервер реализован с помощью фреймворка Flask, а само решение обернуто в Docker контейнер.

* Инструкция
    1. Чтобы собрать докер образ: `docker build -t repo_name/image_name:image_tag .`
    2. Чтобы его запустить: `docker run --rm -p 5000:5000 -v "$PWD/WebServerML/data:/root/WebServerML/data" -v "$PWD/WebServerML/static:/root/WebServerML/static" 
    -i repo_name/image_name`