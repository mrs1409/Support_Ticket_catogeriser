import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse

from predict import predict_ticket
from backend.schemas import TicketRequest, TicketResponse, ModelInfoResponse, HealthResponse

router = APIRouter(prefix="/api", tags=["Classifier"])

# ── HEALTH ──────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse,
            summary="Health check")
async def health():
    try:
        import joblib
        meta = joblib.load(os.path.join(_ROOT, 'models', 'metadata.pkl'))
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=meta.get('best_model_name')
        )
    except Exception:
        return HealthResponse(status="ok", model_loaded=False)

# ── SINGLE TICKET ────────────────────────────────────────────
@router.post("/classify", response_model=TicketResponse,
             summary="Classify a single ticket")
async def classify_single(req: TicketRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Ticket text cannot be empty")
    if len(req.text) > 10000:
        raise HTTPException(status_code=422, detail="Ticket text too long (max 10000 chars)")
    try:
        result = predict_ticket(req.text)
        return TicketResponse(**result)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run python train.py first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ── BATCH CSV ────────────────────────────────────────────────
@router.post("/classify-batch",
             summary="Classify all tickets in a CSV file")
async def classify_batch(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a .csv")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        text_col = None
        for c in ['ticket_text', 'Ticket Description', 'text',
                  'description', 'body', 'content']:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            raise HTTPException(
                status_code=400,
                detail=f"No text column found. Available: {df.columns.tolist()}. "
                       f"Rename your column to 'ticket_text'."
            )

        out_rows = []
        for i, txt in enumerate(df[text_col].fillna('')):
            r = predict_ticket(str(txt))
            out_rows.append({
                'row': i + 1,
                'ticket_preview': str(txt)[:120],
                'predicted_category': r['category'],
                'confidence_pct': f"{r['confidence']*100:.1f}%",
                'urgency': r['urgency'],
                'urgency_reason': r['urgency_reason'],
            })

        out_df = pd.DataFrame(out_rows)
        buf = io.StringIO()
        out_df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    "attachment; filename=classified_tickets.csv"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── MODEL INFO ───────────────────────────────────────────────
@router.get("/model-info",
            summary="Get model performance metrics")
async def model_info():
    try:
        import joblib
        meta = joblib.load(os.path.join(_ROOT, 'models', 'metadata.pkl'))
        comparison = []
        comp_path = os.path.join(_ROOT, 'assets', 'model_comparison.csv')
        if os.path.exists(comp_path):
            comp_df = pd.read_csv(comp_path)
            for _, row in comp_df.iterrows():
                comparison.append({
                    'Model': row['Model'],
                    'Accuracy': round(float(row['Accuracy']), 4),
                    'F1 Macro': round(float(row['F1_Macro']), 4),
                    'F1 Weighted': round(float(row['F1_Weighted']), 4),
                    'Accuracy_pct': f"{float(row['Accuracy'])*100:.2f}%",
                    'F1_Weighted_pct': f"{float(row['F1_Weighted'])*100:.2f}%",
                })
        return {
            'best_model_name': meta.get('best_model_name', 'Unknown'),
            'accuracy': round(float(meta.get('accuracy', 0)), 4),
            'f1_macro': round(float(meta.get('f1_macro', 0)), 4),
            'f1_weighted': round(float(meta.get('f1_weighted', 0)), 4),
            'n_train': int(meta.get('n_train', 0)),
            'n_test': int(meta.get('n_test', 0)),
            'categories': meta.get('labels', []),
            'model_comparison': comparison,
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run python train.py first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── CONFUSION MATRIX IMAGE ───────────────────────────────────
@router.get("/confusion-matrix",
            summary="Get confusion matrix PNG image")
async def confusion_matrix_img():
    path = os.path.join(_ROOT, 'assets', 'confusion_matrix.png')
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Confusion matrix not found. Run python train.py first."
        )
    return FileResponse(path, media_type="image/png",
                       headers={"Cache-Control": "no-cache"})
