from pathlib import Path

from pypdf import PdfReader

from app.services.document_service import DocumentService


service = DocumentService()

pdf_path = Path("uploads/CS338.pdf")

reader = PdfReader(pdf_path)

text = service.extract_text(pdf_path)

print("================================")
print("PDF EXTRACTION TEST")
print("================================")

print("File:", pdf_path)
print("Pages:", len(reader.pages))
print("Characters:", len(text))

for index, page in enumerate(reader.pages):
    page_text = page.extract_text() or ""

    print(
        f"\nPage {index + 1}: "
        f"{len(page_text)} characters"
    )

print("\n================================")
print("FIRST 1000 CHARACTERS")
print("================================")

print(text[:1000])

text = service.extract_text(
    Path("uploads/CS338.pdf")
)

chunks = service.chunk_text(text)

print("Number of chunks:", len(chunks))

for index, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {index} ---")
    print(chunk[:500])