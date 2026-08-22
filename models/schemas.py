from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"
    NONE   = "none"


class CodeSource(BaseModel):
    filepath      : str            = Field(description="Relative path of the file")
    filename      : str            = Field(description="Name of the file")
    language      : str            = Field(description="Programming language")
    function_name : Optional[str]  = Field(description="Function or class name if applicable")
    start_line    : Optional[int]  = Field(description="Starting line number")
    excerpt       : str            = Field(description="Relevant code snippet")


class CodeAnswer(BaseModel):
    answer          : str                = Field(description="Plain English explanation")
    confidence      : ConfidenceLevel    = Field(description="Confidence level")
    sources         : List[CodeSource]   = Field(description="Source files and functions")
    has_code        : bool               = Field(description="Whether answer includes code")
    code_snippet    : Optional[str]      = Field(default=None, description="Key code snippet if relevant")
    language        : Optional[str]      = Field(default=None, description="Language of code snippet")
    warning         : Optional[str]      = Field(default=None, description="Any warning or note")


class RepoInfo(BaseModel):
    repo_name       : str       = Field(description="Name of the repository")
    total_files     : int       = Field(description="Total files processed")
    total_chunks    : int       = Field(description="Total chunks indexed")
    languages       : List[str] = Field(description="Programming languages detected")
    file_tree       : List[str] = Field(description="List of file paths")