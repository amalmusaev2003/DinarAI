class DinarAIException(Exception):
    def __init__(self, message: str = "An error occurred in DinarAI"):
        self.message = message
        super().__init__(self.message)


class ClassificationError(DinarAIException):
    """Exception raised when there's an error in classifying a question."""
    def __init__(self, message: str = "Error classifying question"):
        super().__init__(message)


class TopicClassificationError(ClassificationError):
    """Exception raised when there's an error in classifying a question's topic."""
    def __init__(self, message: str = "Error classifying question topic"):
        super().__init__(message)


class CategoryClassificationError(ClassificationError):
    """Exception raised when there's an error in classifying a question's category."""
    def __init__(self, message: str = "Error classifying question category"):
        super().__init__(message)


class LLMServiceError(DinarAIException):
    """Exception raised when there's an error in the LLM service."""
    def __init__(self, message: str = "Error in LLM service"):
        super().__init__(message)

class WebSearchServiceError(DinarAIException):
    """Exception raised when there's an error in web search service."""
    def __init__(self, message: str = "Error in web search"):
        super().__init__(message)


class VectorStoreError(DinarAIException):
    """Exception raised when there's an error in vector store operations."""
    def __init__(self, message: str = "Error in vector store"):
        super().__init__(message)


class ContextServiceError(DinarAIException):
    """Exception raised when there's an error in context service."""
    def __init__(self, message: str = "Error in context service"):
        super().__init__(message)