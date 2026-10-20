from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AlleleDef(BaseModel):
    function: str
    activity: float | None = None


class Band(BaseModel):
    phenotype: str
    min: float | None = None
    max: float | None = None
    label: str = ""


class PairRule(BaseModel):
    functions: list[str]
    phenotype: str


class GeneDef(BaseModel):
    symbol: str = ""
    aliases: list[str] = Field(default_factory=list)
    model: Literal["activity_score", "function_pair"]
    alleles: dict[str, AlleleDef]
    allele_aliases: dict[str, str] = Field(default_factory=dict)
    reference_allele: str = "*1"
    duplicable: list[str] = Field(default_factory=list)
    bands: list[Band] = Field(default_factory=list)
    pair_rules: list[PairRule] = Field(default_factory=list)

    @property
    def phenotypes(self) -> list[str]:
        names = [b.phenotype for b in self.bands] + [r.phenotype for r in self.pair_rules]
        return list(dict.fromkeys(names))


class DrugDef(BaseModel):
    name: str = ""
    aliases: list[str] = Field(default_factory=list)
    genes: list[str]
    typical_context: str = ""
