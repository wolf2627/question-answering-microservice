from dataclasses import dataclass, replace
from pathlib import Path

from pptx import Presentation
from pypdf import PdfReader

import logging

logger = logging.getLogger(__name__)

# Represents a document that has been loaded from a file
# with its file path and content. 
# It's immutable to ensure data integrity.
@dataclass(frozen=True) 
class LoadedDocument:
    path: Path
    content: str

# Loads all text documents from the specified root directory.
def load_documents(root: Path) -> list[LoadedDocument]:
    documents: list[LoadedDocument] = []

    # Check if the root path exists 
    if not root.exists():
        logger.warning(f"Documents Directory {root} does not exist.")
        print(f"Documents Directory {root} does not exist.")
        return documents
    
    # Iterate and check if each file is supported or not
    for file_path in root.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() not in {'.txt', '.md', '.pdf', '.pptx'}:
            logger.info(f"Skipping unsupported file type: {file_path}")
            continue
        print(f"Processing file: {file_path}")
        text = _read_text(file_path)
        if text.strip():
            documents.append(LoadedDocument(path=file_path, content=text))
        else:
            logger.debug(f"No content extracted from file: {file_path}. Skipping.")
    
    print(f"Loaded {len(documents)} documents.")
    return documents
        
# Reads the text content from a file at the given path.
def _read_text(path: Path)->str:
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        elif suffix == ".pptx":
            presentation = Presentation(str(path))
            texts: list[str] = []
            for slide in presentation.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        texts.append(shape.text)
            return "\n".join(texts)
        else:
            logger.warning(f"Unsupported file type: {path}")
            return ""
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
    return ""


def main():
    print("Starting document ingestion...")
    print(load_documents(Path("./documents")))

if __name__ == "__main__":
    main()