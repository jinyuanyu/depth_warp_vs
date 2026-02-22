# depth_warp_vs/runtime/server/client.py
import requests

if __name__ == "__main__":
    files = {'file': open('assets/test_concat.png','rb')}
    r = requests.post("http://127.0.0.1:8000/infer_concat", files=files, data={"cfg_path":"depth_warp_vs/configs/infer/realtime_512.yaml"})
    print(r.json())
