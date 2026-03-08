"""
AWS Lambda handler — wraps FastAPI with Mangum for Lambda + API Gateway.
"""

from mangum import Mangum
from app.main import app

handler = Mangum(app, lifespan="off")
