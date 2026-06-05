# WeatherAI Smart Agricultural Decision Support System

A **generic, region-agnostic** AI-powered decision support system that combines weather forecasts with tree/forestry data to provide farmers with intelligent planting and harvesting recommendations. The system is designed to work with any crop type and geographic location, with fully configurable parameters.

##  Features

- **Generic Design**: Works with any crop type and geographic location - no hardcoded regions or crops
- **Configurable Parameters**: Crop requirements and risk thresholds can be dynamically customized per region
- **Weather Integration**: Real-time weather data and forecasts from WeatherAI API
- **Tree Analysis**: Canopy health assessment and tree density analysis
- **Predictive Models**: Interpretable ML models (Random Forest, Isolation Forest, KMeans) for optimal planting windows and risk assessment
- **Decision Engine**: AI-powered recommendations for agricultural activities
- **Real-time Processing**: Webhook support for real-time data processing and notifications
- **Firebase Hosting**: Deploy API endpoints to Firebase Functions for scalable cloud hosting
- **REST API**: FastAPI-based backend with automatic documentation

##  Architecture

The system follows SOLID principles and common design patterns:

- **Singleton Pattern**: WeatherAI API client
- **Strategy Pattern**: ML model implementations
- **Chain of Responsibility**: Decision processing pipeline
- **Repository Pattern**: Data access layer
- **Factory Pattern**: Model creation
- **Observer Pattern**: Event notifications

### System Components

```
┌─────────────────┐
│   FastAPI       │
│   Backend       │
└────────┬────────┘
         │
         ├─────────────────────────────────┐
         │                                 │
┌────────▼────────┐              ┌────────▼────────┐
│  Decision       │              │  WeatherAI      │
│  Engine         │◄─────────────│  API Client      │
│  (Chain of      │              │  (Singleton)     │
│   Responsibility)│              └─────────────────┘
└────────┬────────┘
         │
┌────────▼────────┐
│  ML Models      │
│  (Strategy)     │
│  - Planting     │
│  - Risk         │
│  - Canopy       │
└────────┬────────┘
         │
┌────────▼────────┐
│  Data           │
│  Repository     │
│  (Repository)   │
└────────┬────────┘
         │
┌────────▼────────┐
│  Firebase       │
│  Firestore      │
└─────────────────┘
```



##  Quick Start

### Prerequisites

- Python 3.9+
- WeatherAI API key

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd WeatherAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Running the Application

```bash
# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Access the API documentation
# http://localhost:8000/docs
```

## API Endpoints

### Weather Data
- `GET /api/v1/weather/current` - Get current weather conditions
- `GET /api/v1/weather/forecast` - Get weather forecast
- `GET /api/v1/weather/insights` - Get AI-powered weather insights

### Decision Support
- `POST /api/v1/decisions/planting` - Get planting recommendations
- `POST /api/v1/decisions/harvesting` - Get harvesting recommendations
- `POST /api/v1/decisions/risk-assessment` - Get risk assessment
- `GET /api/v1/decisions/history` - Get decision history

### Tree Analysis
- `POST /api/v1/trees/analyze` - Analyze tree canopy and health
- `GET /api/v1/trees/health-trend` - Get canopy health trend analysis
- `GET /api/v1/trees/health-score` - Calculate health score from metrics

### Webhooks (Real-time Processing)
- `POST /api/v1/webhooks/subscribe` - Subscribe to weather trigger events
- `GET /api/v1/webhooks/subscriptions` - List active webhook subscriptions
- `DELETE /api/v1/webhooks/subscriptions/{id}` - Delete webhook subscription
- `POST /api/v1/webhooks/trigger` - Manually trigger webhook event (for testing)
- `POST /api/v1/webhooks/weatherai-callback` - Receive callbacks from WeatherAI

## ML Components

### 1. Planting Window Predictor
Uses weather patterns to predict optimal planting windows based on:
- Temperature trends
- Rainfall probability
- Frost risk
- Soil moisture estimates

### 2. Risk Assessment Model
Evaluates risks for crop damage:
- Frost probability
- Drought likelihood
- Extreme wind events
- Disease risk factors

### 3. Canopy Health Analyzer
Tracks tree health over time:
- Canopy coverage trends
- Health score changes
- Density analysis
- Species identification

##  Generic & Configurable Design

The system is designed to be **region-agnostic** and **crop-agnostic**:

### Dynamic Configuration
- **Crop Requirements**: Crop-specific planting requirements can be updated dynamically via API
- **Risk Thresholds**: Risk assessment thresholds can be customized per region
- **Location Independence**: System works with any geographic location using lat/lon coordinates

### Example: Customizing Crop Requirements
```python
from app.ml.planting_model import PlantingPredictorModel
from app.core.models import CropType

model = PlantingPredictorModel()

# Customize maize requirements for a specific region
model.update_crop_requirements(CropType.MAIZE, {
    "min_temp": 18,  # Higher minimum for warmer region
    "max_temp": 38,
    "optimal_temp": 28,
    "min_soil_moisture": 45,
    "frost_sensitive": True,
    "drought_sensitive": True
})
```

### Example: Customizing Risk Thresholds
```python
from app.ml.risk_model import RiskAssessmentModel

model = RiskAssessmentModel()

# Customize frost thresholds for cold regions
model.update_risk_thresholds("frost", {
    "critical": -5,  # More lenient for cold regions
    "high": -2,
    "medium": 2,
    "low": 5
})
```

##  Configuration

### WeatherAI API
Set your API key in `.env`:
```
WEATHERAI_API_KEY=wai_your_api_key_here
```

##  Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html
```

##  Usage Examples

### Get Planting Recommendations (Generic Location)
```python
import httpx

# Works with any location worldwide
response = httpx.post(
    "http://localhost:8000/api/v1/decisions/planting",
    json={
        "location": {
            "lat": 40.7128,  # New York
            "lon": -74.0060,
            "region": "New York"
        },
        "crop": "maize",
        "planting_date": "2024-06-15"
    }
)
recommendations = response.json()
```

### Subscribe to Webhook for Real-time Alerts
```python
import httpx

# Subscribe to frost alerts for a location
response = httpx.post(
    "http://localhost:8000/api/v1/webhooks/subscribe",
    params={
        "url": "https://your-app.com/webhook",
        "lat": 40.7128,
        "lon": -74.0060,
        "triggers": ["frost", "extreme_wind"],
        "secret": "your-webhook-secret"
    }
)
subscription = response.json()
```

### Analyze Tree Canopy

```python
response = httpx.post(
    "http://localhost:8000/api/v1/trees/analyze",
    files={"image": open("farm.jpg", "rb")},
    data={
        "farmer_id": "F-001",
        "county": "Bomet",
        "land_acres": 2.5
    }
)
analysis = response.json()
```

##  Design Patterns Used

### Singleton Pattern
```python
# WeatherAI API client ensures single instance
weather_client = WeatherAIClient.get_instance()
```

### Strategy Pattern
```python
# Different ML models can be swapped
model = PlantingPredictorModel()
# or
model = RiskAssessmentModel()
```

### Chain of Responsibility
```python
# Decision processing pipeline
decision = DecisionEngine()
decision.add_handler(WeatherValidationHandler())
decision.add_handler(RiskAssessmentHandler())
decision.add_handler(RecommendationHandler())
```

### Repository Pattern
```python
# Abstract data access
repo = FirestoreDecisionRepository()
decision = repo.save(decision)
```

##  Deployment

### Docker Deployment

```bash
# Build image
docker build -t weatherai-decision-system .

# Run container
docker run -p 8000:8000 --env-file .env weatherai-decision-system
```

### Traditional Hosting

```bash
# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production deployment
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

##  Project Structure

```
WeatherAI/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration management
│   ├── api/                    # API endpoints
│   │   ├── __init__.py
│   │   ├── weather.py
│   │   ├── decisions.py
│   │   └── trees.py
│   ├── core/                   # Core business logic
│   │   ├── __init__.py
│   │   ├── weather_client.py   # Singleton API client
│   │   ├── decision_engine.py  # Chain of Responsibility
│   │   └── models.py           # Pydantic models
│   ├── ml/                     # ML components
│   │   ├── __init__.py
│   │   ├── base.py             # Strategy pattern base
│   │   ├── planting_model.py
│   │   ├── risk_model.py
│   │   └── canopy_model.py
│   └── repositories/           # Data access
│       ├── __init__.py
│       ├── base.py
│       └── firestore_repository.py
├── models/                     # Trained ML models
├── tests/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Acknowledgments

- WeatherAI API for providing weather and forestry data
- Firebase for cloud infrastructure
