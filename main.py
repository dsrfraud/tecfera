from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

# from inference.yolo_manager import YoloModelManager
# yolo_manager = YoloModelManager() 

from routers import forgery, passport_verification, face_match, sign_match, face_search
# from routers import bank_statement
# from routers import face_match
from routers import forgery
# FastAPI application instance
app = FastAPI(
    title="Tecfera Backend Service",
    description="Backend APIs for Tecfera document processing and verification",
    contact={
        "name": "Tecfera Solutions",
        "email": "info@tecfera.ai"
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(
#     bank_statement.router,
#     prefix="/bank_statement",
#     tags=["Bank Statement Processing"]
# )

app.include_router(
    forgery.router,
    prefix="",
    tags=["Forgery Detection & QR Processing"]
)

# app.include_router(
#     passport_verification.router,
#     prefix="",
#     tags=["Passport Data Extraction & Validation"]
# )

app.include_router(
    face_match.router,
    prefix="",
    tags=['Face Matching']
)

# app.include_router(
#     face_search.router,
#     prefix="",
#     tags=['Face Repository Search - Including Face Recognition and Training']
# )


app.include_router(
    sign_match.router,
    prefix="",
    tags=['Signature Matching']
)


# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0",port=8000)