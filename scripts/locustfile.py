from locust import HttpUser, task, between

class TritonUser(HttpUser):
    wait_time = between(1, 2)  # Wait 1-2 seconds between tasks

    @task
    def health_check(self):
        """Check if Triton server is healthy"""
        self.client.get("/v2/health/ready", name="Health Check (/v2/health/ready)")

    @task
    def server_metadata(self):
        """Get server metadata"""
        self.client.get("/v2", name="Server Metadata (/v2)")

    @task
    def model_metadata(self):
        """Get model metadata"""
        self.client.get("/v2/models/mobilenetv4", name="Model Metadata (/v2/models/mobilenetv4)")