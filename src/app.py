from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from typing import List
import os

from vector_db import search_VB, dump_to_json, vb_rebuild, INPUT_DIR


app = FastAPI()

if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)


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


@app.get("/rebuild")
def rebuild_endpoint():
    try:
        vb_rebuild()
        return {"message": "База данных успешно перестроена"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Загружаемый файл должен быть в формате PDF")

        file_path = os.path.join(INPUT_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        return {"message": f"Файл {file.filename} успешно загружен в {INPUT_DIR}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))