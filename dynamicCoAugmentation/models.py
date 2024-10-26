from pydantic import BaseModel
from typing import List, Dict

class LiteratureEntity(BaseModel):
    entity_name: List[str]
    entity_type: str
    entity_id: str

class Literature(BaseModel):
    title: str
    abstract: str
    entities: Dict[str, LiteratureEntity]

class RelationExtractionResult(BaseModel):
    entity1: LiteratureEntity
    entity2: LiteratureEntity
    relation: str
    is_generated: bool
