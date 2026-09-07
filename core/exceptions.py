class ExcelliaError(Exception):
    """Base exception for all Excellia RAG errors."""


class ParsingError(ExcelliaError):
    """Raised when a document cannot be parsed."""


class UnsupportedFileTypeError(ExcelliaError):
    """Raised when a file extension is not supported."""


class EmbeddingError(ExcelliaError):
    """Raised when embedding generation fails."""


class VectorStoreError(ExcelliaError):
    """Raised when the vector store is unreachable or a query fails."""


class GenerationError(ExcelliaError):
    """Raised when the LLM backend fails to produce a response."""
