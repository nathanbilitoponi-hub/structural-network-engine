from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

import os
import csv
import json
import uuid
from datetime import datetime
from hashlib import pbkdf2_hmac
from hmac import compare_digest
from base64 import b64encode, b64decode
import secrets

import stripe
import db
from algorithm import run_algorithm
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Structural Network Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_SECRET = os.environ.get("SNE_SESSION_SECRET", "local-dev-secret-change-me")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=False,
)

stripe.api_key = STRIPE_SECRET_KEY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

INDEX_FILE = os.path.join(BASE_DIR, "index.html")
LOGIN_FILE = os.path.join(BASE_DIR, "login.html")
REGISTER_FILE = os.path.join(BASE_DIR, "register.html")
ANALYSIS_FILE = os.path.join(BASE_DIR, "analysis.html")

JOBS_DIR = os.path.join(BASE_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

db.init_db()

CREDIT_PACKAGES = {
    "pack_10": {
        "credits": 10,
        "amount_eur": 10,
        "amount_cents": 1000,
        "currency": "eur",
        "name": "10 credits",
    },
    "pack_50": {
        "credits": 50,
        "amount_eur": 40,
        "amount_cents": 4000,
        "currency": "eur",
        "name": "50 credits",
    },
    "pack_100": {
        "credits": 100,
        "amount_eur": 70,
        "amount_cents": 7000,
        "currency": "eur",
        "name": "100 credits",
    },
}


def utc_now_iso():
    return datetime.utcnow().isoformat() + "Z"


def make_password_hash(password: str):
    salt = secrets.token_bytes(16)
    hashed = pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return b64encode(salt).decode("utf-8"), b64encode(hashed).decode("utf-8")


def verify_password(password: str, salt_b64: str, hash_b64: str):
    salt = b64decode(salt_b64.encode("utf-8"))
    expected_hash = b64decode(hash_b64.encode("utf-8"))
    candidate_hash = pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return compare_digest(candidate_hash, expected_hash)


def get_session_user_id(request: Request):
    return request.session.get("user_id")


def get_current_user(request: Request):
    user_id = get_session_user_id(request)
    if not user_id:
        return None
    return db.fetch_user_by_id(user_id)


def require_user(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def job_dir_path(job_id: str):
    return os.path.join(JOBS_DIR, job_id)


def ensure_job_exists(job_id: str):
    db_job = db.fetch_job(job_id)
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return db_job


def user_can_access_private_job(request: Request, db_job: dict):
    user = get_current_user(request)
    if not user:
        return False
    return db_job["user_id"] == user["id"]


def user_can_access_job_download(request: Request, db_job: dict):
    if db_job["is_public"]:
        return True
    return user_can_access_private_job(request, db_job)


def build_job_response(job_id: str):
    db_job = ensure_job_exists(job_id)

    job_dir = job_dir_path(job_id)
    report_path = os.path.join(job_dir, "report.json")
    graph_path = os.path.join(job_dir, "graph.json")

    if not os.path.isfile(report_path):
        raise HTTPException(status_code=404, detail="Report not found for this job.")

    if not os.path.isfile(graph_path):
        raise HTTPException(status_code=404, detail="Graph not found for this job.")

    with open(report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    owner = db.fetch_user_by_id(db_job["user_id"])

    return {
        "job_id": job_id,
        "user_id": db_job["user_id"],
        "result": report_data.get("result"),
        "created_at": report_data.get("created_at"),
        "input_file": report_data.get("input_file"),
        "metrics": report_data.get("metrics", {}),
        "nodes": graph_data.get("nodes", []),
        "links": graph_data.get("links", []),
        "analysis_url": f"/analysis/{job_id}",
        "downloads": {
            "nodes": f"/download/{job_id}/nodes",
            "edges": f"/download/{job_id}/edges",
            "report": f"/download/{job_id}/report",
        },
        "owner_username": owner["username"] if owner else None,
    }


def write_nodes_csv(file_path, nodes):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "backbone", "degree"])
        for node in nodes:
            writer.writerow([
                node.get("id"),
                node.get("backbone"),
                node.get("degree"),
            ])


def write_edges_csv(file_path, links):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "distance"])
        for edge in links:
            writer.writerow([
                edge.get("source"),
                edge.get("target"),
                edge.get("distance"),
            ])


def write_report_json(report_path, job_id, user_id, created_at, input_file, result):
    payload = {
        "job_id": job_id,
        "user_id": user_id,
        "created_at": created_at,
        "input_file": input_file,
        "result": result.get("result"),
        "metrics": result.get("metrics", {}),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_graph_json(graph_path, nodes, links):
    payload = {
        "nodes": nodes,
        "links": links,
    }
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def download_job_file(request: Request, job_id: str, filename: str, media_type: str):
    db_job = ensure_job_exists(job_id)

    if not user_can_access_job_download(request, db_job):
        raise HTTPException(status_code=403, detail="You do not have access to this file.")

    file_path = os.path.join(job_dir_path(job_id), filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Requested file not found.")

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename,
    )


def get_package_or_400(package_code: str):
    package = CREDIT_PACKAGES.get(package_code)
    if not package:
        raise HTTPException(status_code=400, detail="Invalid credit package.")
    return package


def create_checkout_session_for_package(user_id: int, package_code: str):
    package = get_package_or_400(package_code)

    session = stripe.checkout.Session.create(
        mode="payment",
        client_reference_id=str(user_id),
        success_url=f"{APP_BASE_URL}/?checkout=success",
        cancel_url=f"{APP_BASE_URL}/?checkout=cancel",
        metadata={
            "user_id": str(user_id),
            "package_code": package_code,
            "credits": str(package["credits"]),
        },
        line_items=[
            {
                "price_data": {
                    "currency": package["currency"],
                    "unit_amount": package["amount_cents"],
                    "product_data": {
                        "name": f"Structural Network Engine - {package['name']}",
                        "description": f"{package['credits']} credits",
                    },
                },
                "quantity": 1,
            }
        ],
    )

    db.insert_credit_purchase(
        stripe_session_id=session.id,
        user_id=user_id,
        package_code=package_code,
        credits=package["credits"],
        amount_eur=package["amount_eur"],
        payment_status="pending",
    )

    return session


def fulfill_checkout_session(stripe_session_id, user_id, package_code, credits, amount_eur):
    purchase = db.fetch_credit_purchase_by_session_id(stripe_session_id)

    if purchase and purchase.get("payment_status") == "paid":
        return {"ok": True, "already_fulfilled": True}

    if not purchase:
        db.insert_credit_purchase(
            stripe_session_id=stripe_session_id,
            user_id=user_id,
            package_code=package_code,
            credits=credits,
            amount_eur=amount_eur,
            payment_status="pending",
        )

    purchase = db.fetch_credit_purchase_by_session_id(stripe_session_id)
    if purchase and purchase.get("payment_status") == "paid":
        return {"ok": True, "already_fulfilled": True}

    db.add_credits(user_id, credits)
    db.mark_credit_purchase_completed(stripe_session_id)

    return {"ok": True, "already_fulfilled": False}


def process_checkout_session(session_obj):
    payment_status = session_obj.get("payment_status")
    if payment_status != "paid":
        return

    stripe_session_id = session_obj.get("id")
    if not stripe_session_id:
        return

    metadata = session_obj.get("metadata") or {}
    user_id_str = metadata.get("user_id") or session_obj.get("client_reference_id")
    package_code = metadata.get("package_code")

    if not user_id_str or not package_code:
        return

    package = CREDIT_PACKAGES.get(package_code)
    if not package:
        return

    try:
        user_id = int(user_id_str)
    except Exception:
        return

    user = db.fetch_user_by_id(user_id)
    if not user:
        return

    amount_total = session_obj.get("amount_total")
    if amount_total is not None and int(amount_total) != int(package["amount_cents"]):
        return

    fulfill_checkout_session(
        stripe_session_id=stripe_session_id,
        user_id=user_id,
        package_code=package_code,
        credits=package["credits"],
        amount_eur=package["amount_eur"],
    )


@app.get("/")
async def homepage(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return FileResponse(INDEX_FILE)


@app.get("/login")
def login_page():
    return FileResponse(LOGIN_FILE)


@app.get("/register")
def register_page():
    return FileResponse(REGISTER_FILE)


@app.get("/analysis/{job_id}")
async def analysis_page(job_id: str):
    job = db.fetch_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return FileResponse(ANALYSIS_FILE)


@app.get("/analysis")
async def analysis_root():
    raise HTTPException(status_code=404, detail="Analysis ID missing.")


@app.post("/api/register")
async def api_register(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    username = (body.get("username") or "").strip()
    password = body.get("password") or ""

    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")

    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    existing = db.fetch_user_by_username(username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists.")

    salt_b64, hash_b64 = make_password_hash(password)

    user_id = db.insert_user(
        username=username,
        password_salt=salt_b64,
        password_hash=hash_b64,
        credits=3,
    )

    request.session["user_id"] = user_id
    created_user = db.fetch_user_by_id(user_id)

    return JSONResponse({
        "message": "Registration successful.",
        "user": {
            "id": created_user["id"],
            "username": created_user["username"],
            "credits": created_user["credits"],
        }
    })


@app.post("/api/login")
async def api_login(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    username = (body.get("username") or "").strip()
    password = body.get("password") or ""

    user = db.fetch_user_by_username(username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    if not verify_password(password, user["password_salt"], user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    request.session["user_id"] = user["id"]

    return JSONResponse({
        "message": "Login successful.",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "credits": user["credits"],
        }
    })


@app.post("/api/logout")
def api_logout(request: Request):
    request.session.clear()
    return JSONResponse({"message": "Logout successful."})


@app.get("/api/me")
def api_me(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"authenticated": False})

    return JSONResponse({
        "authenticated": True,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "credits": user["credits"],
        }
    })


@app.get("/api/credit-packages")
def api_credit_packages():
    return JSONResponse({
        "packages": [
            {
                "code": code,
                "credits": package["credits"],
                "amount_eur": package["amount_eur"],
                "amount_cents": package["amount_cents"],
                "currency": package["currency"],
                "name": package["name"],
            }
            for code, package in CREDIT_PACKAGES.items()
        ]
    })


@app.post("/api/create-checkout-session")
async def api_create_checkout_session(request: Request):
    user = require_user(request)

    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe secret key is not configured.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    package_code = (body.get("package_code") or "").strip()
    session = create_checkout_session_for_package(user["id"], package_code)

    return JSONResponse({
        "checkout_url": session.url,
        "session_id": session.id,
        "publishable_key": STRIPE_PUBLISHABLE_KEY,
    })


@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Stripe webhook secret is not configured.")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing Stripe-Signature header.")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=STRIPE_WEBHOOK_SECRET,
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload.")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature.")

    if event["type"] == "checkout.session.completed":
        session_obj = event["data"]["object"]
        process_checkout_session(session_obj)

    return JSONResponse({"received": True})


@app.post("/run")
async def run(request: Request, file: UploadFile = File(...)):
    user = require_user(request)

    credit_result = db.consume_credit(user["id"])
    if not credit_result["ok"]:
        if credit_result["reason"] == "no_credits":
            return JSONResponse(
                status_code=403,
                content={"detail": "No credits remaining. You cannot run a new analysis."}
            )
        return JSONResponse(
            status_code=404,
            content={"detail": "User not found."}
        )

    try:
        job_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        user_id = user["id"]

        job_dir = job_dir_path(job_id)
        os.makedirs(job_dir, exist_ok=True)

        original_filename = file.filename if file.filename else "uploaded_file.csv"
        input_file = os.path.basename(original_filename)
        saved_input_path = os.path.join(job_dir, input_file)

        content = await file.read()
        with open(saved_input_path, "wb") as f:
            f.write(content)

        result = run_algorithm(saved_input_path)

        nodes_path = os.path.join(job_dir, "nodes.csv")
        edges_path = os.path.join(job_dir, "edges.csv")
        report_path = os.path.join(job_dir, "report.json")
        graph_path = os.path.join(job_dir, "graph.json")

        write_nodes_csv(nodes_path, result.get("nodes", []))
        write_edges_csv(edges_path, result.get("links", []))
        write_report_json(
            report_path=report_path,
            job_id=job_id,
            user_id=user_id,
            created_at=created_at,
            input_file=input_file,
            result=result,
        )
        write_graph_json(
            graph_path=graph_path,
            nodes=result.get("nodes", []),
            links=result.get("links", []),
        )

        db.insert_job(
            job_id=job_id,
            user_id=user_id,
            created_at=created_at,
            input_file=input_file,
            is_public=1,
        )

        response = build_job_response(job_id)
        updated_user = db.fetch_user_by_id(user_id)
        response["remaining_credits"] = updated_user["credits"] if updated_user else 0

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "result": "Server error",
                "error": str(e),
            },
        )


@app.get("/jobs")
def list_jobs(request: Request):
    user = require_user(request)
    jobs = db.fetch_jobs_for_user(user["id"], limit=50)

    payload = []
    for job in jobs:
        payload.append({
            "job_id": job["job_id"],
            "user_id": job["user_id"],
            "created_at": job["created_at"],
            "input_file": job["input_file"],
            "analysis_url": f"/analysis/{job['job_id']}",
        })

    return JSONResponse({"jobs": payload})


@app.get("/job/{job_id}")
def get_job(request: Request, job_id: str):
    db_job = ensure_job_exists(job_id)

    if not user_can_access_private_job(request, db_job):
        raise HTTPException(status_code=403, detail="You do not have access to this job.")

    response = build_job_response(job_id)
    user = get_current_user(request)
    response["remaining_credits"] = user["credits"] if user else 0

    return JSONResponse(content=response)


@app.get("/public/job/{job_id}")
def get_public_job(job_id: str):
    db_job = ensure_job_exists(job_id)

    if not db_job["is_public"]:
        raise HTTPException(status_code=403, detail="This analysis is not public.")

    return JSONResponse(content=build_job_response(job_id))


@app.get("/download/{job_id}/nodes")
def download_nodes(request: Request, job_id: str):
    return download_job_file(request, job_id, "nodes.csv", "text/csv")


@app.get("/download/{job_id}/edges")
def download_edges(request: Request, job_id: str):
    return download_job_file(request, job_id, "edges.csv", "text/csv")


@app.get("/download/{job_id}/report")
def download_report(request: Request, job_id: str):
    return download_job_file(request, job_id, "report.json", "application/json")
@app.get("/terms")
def terms(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request})

@app.get("/privacy")
def privacy(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})

@app.get("/refund")
def refund(request: Request):
    return templates.TemplateResponse("refund.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)