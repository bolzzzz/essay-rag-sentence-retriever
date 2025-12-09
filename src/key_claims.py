# This module is designed to use a local LLM for key-claim extraction.
# Currently, it uses a heuristic approach to streamline the workflow.

import re
from typing import List

def split_into_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


async def extract_key_claims_local(essay: str, max_claims: int = 5) -> List[str]:
    
    paras = split_into_paragraphs(essay)
    claims: List[str] = []
    for p in paras:
        parts = [s.strip() for s in re.split(r"(?<=[\.!?])\s+", p) if s.strip()]
        if parts:
            claims.append(parts[0])
        if len(claims) >= max_claims:
            break
    if len(claims) < max_claims:
        strong = []
        for p in paras:
            parts = [s.strip() for s in re.split(r"(?<=[\.!?])\s+", p) if s.strip()]
            for s in parts:
                if re.search(r"\b(should|must|need|prove|argue|claim|show)\b", s, re.IGNORECASE):
                    strong.append(s)
        for s in strong:
            if s not in claims:
                claims.append(s)
            if len(claims) >= max_claims:
                break
    return claims
