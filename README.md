# Gym Muscle Growth API Deployment

A deployment-ready repo folder for the Muscle Growth prediction API using FastAPI and XGBoost.

## What is included

- `Dockerfile` — containerizes the FastAPI application
- `docker-compose.yml` — local development and testing orchestration
- `.dockerignore` — excludes unnecessary files from Docker builds
- `render.yaml` — Render deployment configuration for Docker-based services
- `requirements.txt` — Python dependency file shared from the project root
- `app.py` — production API entrypoint
- `models/muscle_growth_model.pkl`* — model artifact required at runtime
- `feature_names.json`* — expected feature order file for runtime

> *The `models/` folder and artifacts are expected in the project root when deploying.

## Quick Start

### 1. Build locally with Docker

```bash
docker build -t gym-muscle-growth-api .
```

### 2. Run locally with Docker

```bash
docker run -p 8000:8000 gym-muscle-growth-api
```

Then open:

```text
http://localhost:8000/health
```

### 3. Run locally with Docker Compose

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

## Render Deployment

This folder is configured for Render Docker deployment using `render.yaml`.

### How to deploy

1. Push the repository to GitHub.
2. Create a new Web Service on Render.
3. Connect the GitHub repository.
4. Choose Docker as the environment.
5. Use the default `Dockerfile` and deploy.

### Recommended Render settings

- Environment: Docker
- Dockerfile path: `Dockerfile`
- Automatic deploys: enabled
- Health check path: `/health`

## API Contract

### Health check

```http
GET /health
```

### Predict endpoint

```http
POST /predict?user_id=anon
Content-Type: application/json
```

Example payload:

```json
{
  "sleep_hours": 7.5,
  "training_frequency_per_week": 4,
  "calorie_surplus": 300,
  "progressive_overload_score": 6,
  "training_experience_years": 2,
  "body_fat_percentage": 18,
  "protein_intake_g": 180,
  "stress_level": 3
}
```

Response includes:

- `probability`: muscle growth likelihood score
- `label`: binary prediction (0 / 1)
- `message`: human-friendly guidance
- `advice`: top coaching recommendations
- `delta_messages`: recent input changes summary
- `feature_order`: input feature order used for prediction

## Notes

- Ensure `models/muscle_growth_model.pkl` and `feature_names.json` exist in the repo root when building or deploying.
- The `MODEL_PATH` config defaults to `models/muscle_growth_model.pkl`.
- The `FEATURES_PATH` config defaults to `feature_names.json`.

## Troubleshooting

- If the container fails on startup, confirm the model and feature files are present and readable.
- If dependency installation fails, verify `requirements.txt` is in the same folder as `Dockerfile`.
- Use `docker logs <container-id>` to inspect runtime errors.

## Contact

For repo support or questions, update the GitHub repo description with usage notes and deployment details.
