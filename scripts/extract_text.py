import fitz               # PyMuPDF 1.25.5
import pytesseract
from PIL import Image
import io, sys, json


def extract_text_and_images_in_order(pdf_path: str) -> str:
    """
    Извлекает PDF-текст и OCR-текст изображений в порядке чтения.
    Добавляет заголовки:
      --- Page N ---               для каждого текстового блока
      --- OCR Text from Image on Page N ---  для каждой картинки
    Возвращает одну большую строку.
    """
    doc = fitz.open(pdf_path)
    output_blocks = []  # собираем фрагменты сюда

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)

        try:
            page_dict = page.get_text("dict", sort=True)
        except Exception:
            # если чтение страницы упало, пропускаем её
            continue

        blocks = page_dict.get("blocks", [])

        for block in blocks:
            btype = block.get("type")

            # -------- текстовый блок --------
            if btype == 0:
                text_content = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text_content += span.get("text", "")
                    text_content += "\n"
                if text_content.strip():
                    output_blocks.append(
                        f"\n\n--- Page {page_index + 1} ---\n{text_content}"
                    )

            # -------- блок-картинка --------
            elif btype == 1:
                image_bytes = block.get("image")  # байты PNG/JPEG
                if image_bytes:
                    try:
                        img = Image.open(io.BytesIO(image_bytes))
                        # rus+eng — русские + латиница
                        ocr_text = pytesseract.image_to_string(
                            img, lang="rus+eng"
                        ).strip()
                    except Exception:
                        ocr_text = ""

                    if ocr_text:
                        output_blocks.append(
                            f"\n\n--- OCR Text from Image on Page {page_index + 1} ---\n{ocr_text}"
                        )

    return "".join(output_blocks)


# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_text.py <PDF_file_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    extracted = extract_text_and_images_in_order(pdf_path)
    print(json.dumps([{"text": extracted}], ensure_ascii=False))

