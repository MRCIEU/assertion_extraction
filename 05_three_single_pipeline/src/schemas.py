from pydantic import BaseModel
from typing import List, Optional

class Sentence(BaseModel):
    id: int
    text: str
    is_finding: bool

class FindingDoc(BaseModel):
    pmid: str
    finding_ids: List[int]

class PartialResolved(BaseModel):
    id: int
    resolved: str

class ResolvedSentence(BaseModel):
    id: int
    original: str
    resolved: str

class CompletionDoc(BaseModel):
    pmid: str
    sentences: List[ResolvedSentence]

class Assertion(BaseModel):
    id: int
    subject: str
    predicate: str
    object: str
    condition: Optional[str] = None

class AssertionDoc(BaseModel):
    pmid: str
    assertion: List[Assertion]