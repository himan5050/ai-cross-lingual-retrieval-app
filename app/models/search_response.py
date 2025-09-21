from pydantic import BaseModel # pyright: ignore[reportMissingImports]


class SearchResponse(BaseModel):
    id: str
    title: str
    text: str
    lang: str
    score: float