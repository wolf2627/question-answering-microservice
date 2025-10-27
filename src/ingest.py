from dataclasses import dataclass, replace
from pathlib import Path

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
        if file_path.is_file() and file_path.suffix.lower() not in {'.txt', '.md', '.pdf'}:
            logger.info(f"Skipping unsupported file type: {file_path}")
            continue
        # TODO: Read and process the file content
        print(f"Processing file: {file_path}")
        
    print(f"Loaded {len(documents)} documents.")


def main():
    print("Starting document ingestion...")
    load_documents(Path("./documents"))

if __name__ == "__main__":
    main()