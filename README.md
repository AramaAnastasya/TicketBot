## Установка и настройка проекта

Для корректной работы проекта необходима версия Python 3.9.

### Шаги по установке:

1. **Создайте и активируйте виртуальную среду:**
    ```bash
    py -m venv env
    env/Scripts/activate
    ```

2. **Установите все библиотеки из файла `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Добавьте модели Ollama:**
    ```bash
    ollama pull mxbai-embed-large
    ollama run llama3.2:3b
    ```

4. **Установите Tesseract OCR:**
    * Скачайте файл `rus.traineddata` с [GitHub](https://github.com/tesseract-ocr/tessdata).
    * Поместите файл в директорию `C:\Program Files\Tesseract-OCR\tessdata`, чтобы Tesseract мог обрабатывать изображения на русском языке.

