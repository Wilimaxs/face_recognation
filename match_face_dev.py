import base64
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel
import face_recognition
import numpy as np
import uvicorn
import cv2
import io
import uuid
import json
from firebase_admin import credentials, firestore, initialize_app, storage

# Load env
load_dotenv()

# Decode base64 credential
base64_str = os.getenv("FIREBASE_CREDENTIALS_BASE64")
if not base64_str:
    raise Exception("Missing FIREBASE_CREDENTIALS_BASE64 in environment")

cred_dict = json.loads(base64.b64decode(base64_str))

# Inisialisasi Firebase
cred = credentials.Certificate(cred_dict)  # ganti path dengan file key-mu
initialize_app(cred)
db = firestore.client()

# Inisialisasi FastAPI
app = FastAPI()

# Middleware untuk mengizinkan akses dari Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint tes
@app.get("/")
def root():
    return {"message": "Face Recognition API ready"}

# Endpoint untuk pencocokan wajah
@app.post("/match-face")
async def match_face(file: UploadFile = File(...), doc_id: str = Form(...)):
    try:
        # Baca dan proses gambar
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Deteksi wajah
        face_locations = face_recognition.face_locations(rgb_img)
        if not face_locations:
            return {"match": False, "error": "❌ Tidak ada wajah terdeteksi"}

        face_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]

        # Ambil encoding dari dokumen Firestore berdasarkan doc_id
        doc_ref = db.collection("face_user").document(doc_id)
        doc = doc_ref.get()

        if not doc.exists:
            return {"match": False, "error": f"⚠️ Tidak ditemukan data dengan ID '{doc_id}'"}

        data = doc.to_dict()
        known_encoding = np.array(data["encoding"], dtype=np.float32)
        known_name = data["name"]

        # Bandingkan
        match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.5)[0]
        distance = face_recognition.face_distance([known_encoding], face_encoding)[0]

        if match:
            return {
                "match": True,
                "name": known_name,
                "similarity": float(1 - distance),
                "doc_id": doc_id
            }
        else:
            return {"match": False, "similarity": float(1 - distance), "doc_id": doc_id}

    except Exception as e:
        return JSONResponse(status_code=500, content={"match": False, "error": str(e)})


@app.post("/register-face")
async def register_face(
    file: UploadFile = File(...),
    name: str = Form(...),
    doc_id: str = Form(...)  # locker1 / locker2 / locker3
):
    try:
        # Baca gambar dari UploadFile
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Deteksi wajah
        face_locations = face_recognition.face_locations(rgb_img)
        if not face_locations:
            return JSONResponse(status_code=400, content={"success": False, "error": "❌ Tidak ada wajah terdeteksi"})

        # Ambil encoding dari wajah pertama
        encoding = face_recognition.face_encodings(rgb_img, face_locations)[0].tolist()

        # Cek apakah doc_id sudah ada
        doc_ref = db.collection("face_user").document(doc_id)
        doc = doc_ref.get()

        # Simpan atau update
        doc_ref.set({
            "name": name,
            "encoding": encoding
        })

        action = "diupdate" if doc.exists else "dibuat baru"

        return {
            "success": True,
            "message": f"Wajah berhasil {action} di Firestore",
            "doc_id": doc_id
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})