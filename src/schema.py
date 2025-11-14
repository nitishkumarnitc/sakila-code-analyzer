# src/schema.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Method(BaseModel):
    name: str
    signature: str
    description: Optional[str] = None
    complexity_notes: Optional[str] = None

class Module(BaseModel):
    module_name: str
    path: str
    description: Optional[str] = None
    key_methods: List[Method]

class ProjectKnowledge(BaseModel):
    project_name: str
    project_overview: str
    primary_languages: List[str]
    key_modules: List[Module]
    global_complexity_notes: Optional[str] = None
    assumptions: Optional[List[str]] = None
    generated_at: datetime
