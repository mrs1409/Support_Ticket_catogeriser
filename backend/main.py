from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers.classify import router

app = FastAPI(
    title="TicketSense — Support Ticket Classifier API",
    description="""
## NLP-powered IT service-desk ticket classification

### Features
- **Category prediction** — classifies tickets into 8 real IT categories:
  Access, Hardware, HR Support, Storage, Purchase, Administrative rights,
  Internal Project, Miscellaneous
- **Urgency detection** — High / Medium / Low using weighted keyword rules
- **Confidence scores** — probability breakdown across all 8 categories
- **Batch processing** — upload a CSV, download classified results

### Model
Trained on 47,800+ real IT service-desk tickets (Kaggle) + casual-language
augmentation. Best of Logistic Regression / Naive Bayes / LinearSVC
auto-selected by F1 weighted score. Training accuracy: 86.48%.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",                          # Vercel production + any domain
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,          # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Support Ticket Classifier API",
        "docs": "/docs",
        "health": "/api/health",
        "classify": "/api/classify",
        "model_info": "/api/model-info",
    }
