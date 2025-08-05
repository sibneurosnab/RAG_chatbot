from flask import Flask, request, jsonify
import subprocess, os, tempfile, uuid

app = Flask(__name__)

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/process_pdf", methods=["POST"])
def process_pdf():
    """Обрабатывает PDF, переданный через POST запрос."""
    f = request.files["pdf"]
    pdf_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.pdf")
    f.save(pdf_path)

    try:
        # Запускаем внешний скрипт для извлечения текста из PDF
        result = subprocess.check_output(
            ["python3", "/home/node/scripts/extract_text.py", pdf_path],
            stderr=subprocess.STDOUT,
            text=True
        )
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "extract_text failed", "msg": e.output}), 500
    finally:
        os.remove(pdf_path)  # Удаляем временный файл после обработки

    return jsonify({"text": result})  # Возвращаем извлечённый текст

if __name__ == "__main__":  # Заменил 'name' на '__name__'
    app.run(host="0.0.0.0", port=5000, debug=False)  # Запускаем Flask на 5000 порту