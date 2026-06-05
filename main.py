"""
Firebase Cloud Functions entry point for WeatherAI API.
This file is used for Firebase Functions deployment.
"""

import json
from firebase_functions import https_fn, options
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import the main app
from app.main import app as fastapi_app

# Configure CORS for Firebase Functions
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firebase Function specification
@https_fn.on_request(
    cors=options.CorsOptions(
        cors_origins=["*"],
        cors_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    ),
    region="us-central1",
    memory=options.MemoryOption.MB_512,
    cpu=options.CpuOption.MICRO,
    timeout_sec=60
)
def api(req: https_fn.Request) -> https_fn.Response:
    """
    Firebase Cloud Function that handles all API requests.
    Routes requests to the FastAPI application.
    """
    # Convert Firebase request to ASGI scope
    scope = {
        "type": "http",
        "method": req.method,
        "path": req.path,
        "query_string": req.query_string.decode("utf-8") if req.query_string else "",
        "headers": dict(req.headers),
        "server": ("firebase-functions", 80),
        "scheme": "https",
    }
    
    # Create ASGI receive/send channels
    async def receive():
        return {"type": "http.request", "body": req.data}
    
    async def send(message):
        if message["type"] == "http.response.start":
            status = message["status"]
            headers = message["headers"]
            return https_fn.Response(
                body=None,
                status=status,
                headers=dict(headers)
            )
        elif message["type"] == "http.response.body":
            body = message.get("body", b"")
            return https_fn.Response(
                body=body,
                status=200,
                headers={"Content-Type": "application/json"}
            )
    
    # Call the FastAPI app
    from asgiref.wsgi import WsgiToAsgi
    asgi_app = WsgiToAsgi(fastapi_app)
    
    # Process the request through FastAPI
    response = fastapi_app.handle_request(scope, receive, send)
    
    return response
