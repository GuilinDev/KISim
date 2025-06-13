from locust import HttpUser, task, between, events
import numpy as np
import time
import json
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TritonUser(HttpUser):
    """
    Locust user class for testing Triton Inference Server.
    Includes both health/metadata checks and model inference requests.
    """
    wait_time = between(1, 2)  # Wait 1-2 seconds between tasks

    def on_start(self):
        """Initialize user session"""
        # Generate a synthetic image once and reuse it for all inference requests
        # This avoids generating a new image for each request, which can be CPU-intensive
        self.synthetic_image = self.generate_synthetic_image()
        logger.info("User session started, synthetic image generated")

    def generate_synthetic_image(self):
        """Generate a synthetic image for inference"""
        # Generate random image data with shape [1, 3, 224, 224] (batch, channels, height, width)
        return np.random.rand(1, 3, 224, 224).astype(np.float32)

    @task(1)
    def health_check(self):
        """Check if Triton server is healthy"""
        with self.client.get("/v2/health/ready", name="Health Check (/v2/health/ready)", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    def server_metadata(self):
        """Get server metadata"""
        with self.client.get("/v2", name="Server Metadata (/v2)", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Server metadata request failed: {response.status_code}")

    @task(1)
    def model_metadata(self):
        """Get model metadata"""
        with self.client.get("/v2/models/mobilenetv4", name="Model Metadata (/v2/models/mobilenetv4)", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Model metadata request failed: {response.status_code}")

    @task(5)  # Higher weight for inference tasks
    def model_inference(self):
        """Send inference request to the model"""
        # Create request payload using the pre-generated synthetic image
        payload = {
            "inputs": [
                {
                    "name": "pixel_values",
                    "shape": [1, 3, 224, 224],
                    "datatype": "FP32",
                    "data": self.synthetic_image.flatten().tolist()
                }
            ]
        }

        # Send POST request to the inference endpoint
        start_time = time.time()
        with self.client.post(
            "/v2/models/mobilenetv4/infer",
            json=payload,
            name="Model Inference (/v2/models/mobilenetv4/infer)",
            catch_response=True
        ) as response:
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds

            if response.status_code == 200:
                try:
                    # Parse response to extract predictions
                    response_data = response.json()

                    # Find the output tensor (usually named "logits" or similar)
                    output_data = None
                    for output in response_data.get('outputs', []):
                        if output.get('name') in ['logits', 'output', 'predictions']:
                            output_data = np.array(output.get('data')).reshape(output.get('shape'))
                            break

                    if output_data is not None:
                        # Get top predicted class
                        top_class = np.argmax(output_data[0])
                        response.success()
                        # Log extra info that can be useful for analysis
                        response.metadata = {
                            "top_class": int(top_class),
                            "latency_ms": latency
                        }
                    else:
                        response.failure("Could not find output tensor in response")
                except Exception as e:
                    response.failure(f"Error parsing response: {str(e)}")
            else:
                response.failure(f"Inference request failed: {response.status_code} - {response.text}")

# Optional: Add custom event handlers for collecting additional metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Log additional information about requests"""
    if exception:
        logger.error(f"Request {name} failed: {exception}")
    elif hasattr(kwargs.get('response', {}), 'metadata'):
        metadata = kwargs['response'].metadata
        logger.debug(f"Request {name} completed in {response_time}ms, top class: {metadata.get('top_class')}")
