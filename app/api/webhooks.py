"""Webhook API endpoints for real-time data processing"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from typing import Optional, List
import logging
from datetime import datetime
import hashlib

from ..core.weather_client import get_weather_client
from ..core.decision_engine import DecisionEngine
from ..core.models import (
    Location,
    CropType,
    PlantingDecisionRequest,
    HarvestingDecisionRequest
)
from ..ml.planting_model import PlantingPredictorModel
from ..ml.risk_model import RiskAssessmentModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/webhooks", tags=["webhooks"])

# Initialize components
planting_model = PlantingPredictorModel()
risk_model = RiskAssessmentModel()

# Webhook storage (in production, use database)
webhook_subscriptions = {}


class WebhookSubscription:
    """Webhook subscription model"""
    def __init__(
        self,
        url: str,
        location: Location,
        triggers: List[str],
        secret: Optional[str] = None
    ):
        self.url = url
        self.location = location
        self.triggers = triggers
        self.secret = secret
        self.active = True
        self.created_at = datetime.utcnow()


def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Verify webhook signature for security.
    
    Args:
        payload: Request payload
        signature: Received signature
        secret: Webhook secret
    
    Returns:
        True if signature is valid
    """
    expected_signature = hashlib.sha256(f"{payload}{secret}".encode()).hexdigest()
    return signature == expected_signature


@router.post("/subscribe")
async def subscribe_to_webhook(
    url: str,
    lat: float,
    lon: float,
    triggers: List[str],
    secret: Optional[str] = None,
    region: Optional[str] = None,
    country: Optional[str] = None
):
    """
    Subscribe to weather trigger events via webhook.
    WeatherAI will POST to your URL when conditions are met.
    
    Args:
        url: Your webhook endpoint URL
        lat: Latitude for monitoring
        lon: Longitude for monitoring
        triggers: List of trigger types (rain, extreme_wind, frost, drought)
        secret: Optional secret for signature verification
        region: Optional region name
        country: Optional country name
    
    Returns:
        Subscription confirmation
    """
    try:
        location = Location(
            lat=lat,
            lon=lon,
            region=region,
            country=country
        )
        
        subscription_id = f"webhook_{datetime.utcnow().timestamp()}"
        subscription = WebhookSubscription(
            url=url,
            location=location,
            triggers=triggers,
            secret=secret
        )
        
        webhook_subscriptions[subscription_id] = subscription
        
        logger.info(f"Webhook subscription created: {subscription_id}")
        
        return {
            "subscription_id": subscription_id,
            "url": url,
            "location": location.dict(),
            "triggers": triggers,
            "active": True,
            "created_at": subscription.created_at.isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error creating webhook subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/subscriptions")
async def list_webhook_subscriptions():
    """
    List all active webhook subscriptions.
    
    Returns:
        List of webhook subscriptions
    """
    subscriptions = []
    for sub_id, sub in webhook_subscriptions.items():
        if sub.active:
            subscriptions.append({
                "subscription_id": sub_id,
                "url": sub.url,
                "location": sub.location.dict(),
                "triggers": sub.triggers,
                "active": sub.active,
                "created_at": sub.created_at.isoformat()
            })
    
    return {"webhooks": subscriptions}


@router.delete("/subscriptions/{subscription_id}")
async def delete_webhook_subscription(subscription_id: str):
    """
    Delete a webhook subscription.
    
    Args:
        subscription_id: ID of subscription to delete
    
    Returns:
        Deletion confirmation
    """
    if subscription_id not in webhook_subscriptions:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    webhook_subscriptions[subscription_id].active = False
    del webhook_subscriptions[subscription_id]
    
    logger.info(f"Webhook subscription deleted: {subscription_id}")
    
    return {"message": "Subscription deleted successfully"}


@router.post("/trigger")
async def trigger_webhook_event(
    trigger_type: str,
    location: Location,
    weather_data: dict,
    background_tasks: BackgroundTasks
):
    """
    Manually trigger a webhook event (for testing).
    In production, this would be called by WeatherAI when conditions are met.
    
    Args:
        trigger_type: Type of trigger (rain, frost, etc.)
        location: Location where trigger occurred
        weather_data: Current weather data
        background_tasks: FastAPI background tasks
    
    Returns:
        Trigger confirmation
    """
    # Find matching subscriptions
    matching_subscriptions = []
    
    for sub_id, sub in webhook_subscriptions.items():
        if not sub.active:
            continue
        
        # Check location proximity (within 0.1 degrees)
        lat_diff = abs(sub.location.lat - location.lat)
        lon_diff = abs(sub.location.lon - location.lon)
        
        if lat_diff < 0.1 and lon_diff < 0.1:
            # Check trigger type
            if trigger_type in sub.triggers:
                matching_subscriptions.append(sub)
    
    # Send webhooks in background
    for subscription in matching_subscriptions:
        background_tasks.add_task(
            send_webhook_notification,
            subscription.url,
            trigger_type,
            location,
            weather_data,
            subscription.secret
        )
    
    return {
        "message": f"Triggered {len(matching_subscriptions)} webhooks",
        "trigger_type": trigger_type,
        "subscriptions_triggered": len(matching_subscriptions)
    }


async def send_webhook_notification(
    url: str,
    trigger_type: str,
    location: Location,
    weather_data: dict,
    secret: Optional[str] = None
):
    """
    Send webhook notification to subscriber.
    
    Args:
        url: Webhook URL
        trigger_type: Type of trigger
        location: Location
        weather_data: Weather data
        secret: Optional secret for signature
    """
    import httpx
    
    payload = {
        "trigger_type": trigger_type,
        "location": location.dict(),
        "weather_data": weather_data,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    headers = {"Content-Type": "application/json"}
    
    if secret:
        signature = hashlib.sha256(
            f"{str(payload)}{secret}".encode()
        ).hexdigest()
        headers["X-Webhook-Signature"] = signature
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=10)
            logger.info(f"Webhook sent to {url}: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to send webhook to {url}: {e}")


@router.post("/weatherai-callback")
async def weatherai_callback(request: Request, background_tasks: BackgroundTasks):
    """
    Receive callbacks from WeatherAI when weather conditions match triggers.
    This endpoint would be registered with WeatherAI's webhook system.
    
    Args:
        request: Incoming request from WeatherAI
        background_tasks: FastAPI background tasks
    
    Returns:
        Acknowledgment
    """
    try:
        payload = await request.json()
        
        trigger_type = payload.get("trigger_type")
        location_data = payload.get("location", {})
        weather_data = payload.get("weather_data", {})
        
        location = Location(
            lat=location_data.get("lat", 0),
            lon=location_data.get("lon", 0),
            region=location_data.get("region"),
            country=location_data.get("country")
        )
        
        # Trigger matching webhooks
        await trigger_webhook_event(
            trigger_type=trigger_type,
            location=location,
            weather_data=weather_data,
            background_tasks=background_tasks
        )
        
        return {"message": "Callback received and processed"}
    
    except Exception as e:
        logger.error(f"Error processing WeatherAI callback: {e}")
        raise HTTPException(status_code=500, detail=str(e))
