from fastapi import FastAPI, HTTPException, Query
from typing import List

from vector_db import search_VB, dump_to_json
app = FastAPI()

@app.get("/search")
def search_endpoint(
    query: str = Query(..., description="Поисковый запрос"),
    k: int = Query(5, ge=1, description="Количество результатов"),
    threshold: float = Query(0.87, ge=0.0, le=1.0, description="Пороговая оценка")
):
    try:
        documents = [dump_to_json(doc, i) for i, doc in enumerate(search_VB(query, k=5))]

        if not documents:
            raise HTTPException(status_code=404, detail="Нет релевантных документов")
        return {"results": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
