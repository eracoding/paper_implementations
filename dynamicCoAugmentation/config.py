from pydantic import BaseSettings, Field
from typing import List

class APIConfig(BaseSettings):
    api_key: str = Field(..., env="API_KEY")
    years: List[int] = Field(default_factory=lambda: list(range(2011, 2021)))
    output_path: str = "../extracted"
    base_path: str = "by_year"
    max_workers: int = 4
    request_timeout: int = 10

    class Config:
        env_file = ".env"
