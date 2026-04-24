# app.py
# FastAPI REST API for the multi-symptom NER detector
# Exposes a single POST endpoint /predict that accepts a user query
# and returns detected entities with negation flags
#
# To run: uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger
import time

from model.ner_pipeline import predict

app = FastAPI(
    title="Multi-Symptom Detector API",
    description="NER-based API to detect symptoms, duration, severity from medical queries",
    version="1.0.0",
)


# ── Request / Response schemas ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        example="I have fever, headache and knee pain from 3 days",
    )


class BulkQueryRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        example=[
            "I have fever and headache from 3 days",
            "Patient with diabetes c/o severe chest pain",
            "No fever but experiencing cough and fatigue"
        ]
    )


class EntityResult(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int
    negated: bool


class PredictResponse(BaseModel):
    original_text: str
    processed_text: str
    entities: list[EntityResult]
    processing_time_ms: float


class BulkPredictResponse(BaseModel):
    results: list[PredictResponse]
    total_texts: int
    total_processing_time_ms: float
    average_time_per_text_ms: float


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "multi-symptom-detector"}


@app.get("/health", tags=["Health"])
def health():
    """Detailed health check."""
    return {"status": "healthy", "model": "loaded"}


@app.post("/predict", response_model=PredictResponse, tags=["NER"])
def predict_entities(request: QueryRequest):
    """
    Detect medical entities from a user query.

    - **text**: The raw user input (e.g., "I have fever and headache from 3 days")

    Returns detected entities with:
    - **text**: the entity text
    - **label**: SYMPTOM / INDICATION / SEVERITY / DURATION
    - **negated**: True if the entity is negated (e.g., "no fever")
    """
    try:
        start = time.time()
        result = predict(request.text)
        elapsed_ms = round((time.time() - start) * 1000, 2)

        logger.info(f"Query: '{request.text}' | Entities found: {len(result['entities'])} | Time: {elapsed_ms}ms")

        return PredictResponse(
            original_text=result["original_text"],
            processed_text=result["processed_text"],
            entities=result["entities"],
            processing_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/bulk", response_model=BulkPredictResponse, tags=["NER"])
def predict_bulk(request: BulkQueryRequest):
    """
    Detect medical entities from multiple queries in bulk.

    - **texts**: List of text queries (max 100)

    Returns:
    - **results**: List of predictions for each text
    - **total_texts**: Number of texts processed
    - **total_processing_time_ms**: Total time taken
    - **average_time_per_text_ms**: Average time per text

    Example request:
    ```json
    {
      "texts": [
        "I have fever and headache from 3 days",
        "Patient with diabetes c/o severe chest pain",
        "No fever but experiencing cough and fatigue"
      ]
    }
    ```
    """
    try:
        start_time = time.time()
        results = []
        
        for text in request.texts:
            text_start = time.time()
            result = predict(text)
            text_elapsed_ms = round((time.time() - text_start) * 1000, 2)
            
            results.append(PredictResponse(
                original_text=result["original_text"],
                processed_text=result["processed_text"],
                entities=result["entities"],
                processing_time_ms=text_elapsed_ms,
            ))
        
        total_time_ms = round((time.time() - start_time) * 1000, 2)
        avg_time_ms = round(total_time_ms / len(request.texts), 2)
        
        logger.info(f"Bulk prediction: {len(request.texts)} texts | Total time: {total_time_ms}ms | Avg: {avg_time_ms}ms")
        
        return BulkPredictResponse(
            results=results,
            total_texts=len(request.texts),
            total_processing_time_ms=total_time_ms,
            average_time_per_text_ms=avg_time_ms,
        )
    
    except Exception as e:
        logger.error(f"Bulk prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
