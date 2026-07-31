import time
import requests

class VisionRestoreClient:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def upload_image(self, path):
        with open(path, "rb") as f:
            return requests.post(f"{self.base_url}/api/v1/images/upload", files={"file": f}).json()["data"]

    def analyze_image(self, path):
        with open(path, "rb") as f:
            return requests.post(f"{self.base_url}/api/v1/images/analyze", files={"file": f}).json()["data"]

    def create_task(self, image_id, user_goal="", mode="auto", priority="balanced", model_id=None):
        payload = {"image_id": image_id, "user_goal": user_goal, "mode": mode, "priority": priority, "model_id": model_id}
        return requests.post(f"{self.base_url}/api/v1/tasks", json=payload).json()["data"]

    def get_task_status(self, task_id):
        return requests.get(f"{self.base_url}/api/v1/tasks/{task_id}").json()["data"]

    def cancel_task(self, task_id):
        return requests.post(f"{self.base_url}/api/v1/tasks/{task_id}/cancel").json()["data"]

    def get_results(self, task_id):
        return requests.get(f"{self.base_url}/api/v1/tasks/{task_id}/results").json()["data"]

    def download_result(self, file_id, output_path):
        r = requests.get(f"{self.base_url}/api/v1/files/{file_id}")
        r.raise_for_status()
        open(output_path, "wb").write(r.content)

    def download_report(self, task_id, output_path):
        r = requests.get(f"{self.base_url}/api/v1/tasks/{task_id}/report")
        r.raise_for_status()
        open(output_path, "wb").write(r.content)

    def list_models(self):
        return requests.get(f"{self.base_url}/api/v1/models").json()["data"]

if __name__ == "__main__":
    c = VisionRestoreClient()
    print(c.list_models())
