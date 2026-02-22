# udp_stream_server.py
import argparse
import socket
import struct
import time
import threading
import queue
import cv2
import numpy as np

try:
    from turbojpeg import TurboJPEG
    _jpeg = TurboJPEG()
    def jpeg_decode(b):
        return _jpeg.decode(b)
except Exception:
    _jpeg = None
    def jpeg_decode(b):
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("cv2.imdecode failed")
        return img

MAGIC = b'IM'
VERSION = 1
HEADER_STRUCT = struct.Struct('!2s B B I Q H H H')
HEADER_SIZE = HEADER_STRUCT.size

class FrameAssembly:
    __slots__ = ("created_time", "timestamp_us", "total_chunks", "chunks", "received", "size")
    def __init__(self, total_chunks, timestamp_us):
        self.created_time = time.monotonic()
        self.timestamp_us = timestamp_us
        self.total_chunks = total_chunks
        self.chunks = [None] * total_chunks
        self.received = 0
        self.size = 0

    def add(self, chunk_id, payload):
        if 0 <= chunk_id < self.total_chunks and self.chunks[chunk_id] is None:
            self.chunks[chunk_id] = bytes(payload)
            self.received += 1
            self.size += len(payload)
            return self.received == self.total_chunks
        return False

    def build(self):
        return b"".join(self.chunks)

class Reassembler:
    def __init__(self, max_age=0.5, max_frames=128):
        self.map = {}  # key: (stream, frame_id) -> FrameAssembly
        self.max_age = max_age
        self.max_frames = max_frames

    def add_chunk(self, stream, frame_id, timestamp_us, total_chunks, chunk_id, payload):
        key = (stream, frame_id)
        fa = self.map.get(key)
        if fa is None:
            # bound size
            if len(self.map) > self.max_frames:
                self.gc()
            fa = FrameAssembly(total_chunks, timestamp_us)
            self.map[key] = fa
        # If total_chunks mismatched (e.g. duplicate frame_id collision), reset
        if fa.total_chunks != total_chunks:
            fa = FrameAssembly(total_chunks, timestamp_us)
            self.map[key] = fa
        done = fa.add(chunk_id, payload)
        if done:
            data = fa.build()
            del self.map[key]
            return data, fa.timestamp_us, frame_id, stream
        return None

    def gc(self):
        now = time.monotonic()
        to_del = [k for k, v in self.map.items() if now - v.created_time > self.max_age]
        for k in to_del:
            del self.map[k]

def run_receiver(bind_ip, port, out_queues):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind_ip, port))
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    except Exception:
        pass
    print(f"Server listening on {bind_ip}:{port}")

    reasm = Reassembler(max_age=0.5, max_frames=512)
    last_gc = time.monotonic()
    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except Exception:
            break
        if len(data) < HEADER_SIZE:
            continue
        try:
            magic, ver, stream, frame_id, ts_us, total_chunks, chunk_id, payload_len = HEADER_STRUCT.unpack_from(data, 0)
        except struct.error:
            continue
        if magic != MAGIC or ver != VERSION:
            continue
        payload = data[HEADER_SIZE:]
        if len(payload) != payload_len:
            continue
        res = reasm.add_chunk(stream, frame_id, ts_us, total_chunks, chunk_id, payload)
        if res is not None:
            frame_bytes, timestamp_us, fid, sid = res
            q = out_queues.get(sid)
            if q is not None:
                # Drop if queue full to keep latency low
                try:
                    q.put_nowait((fid, timestamp_us, frame_bytes))
                except queue.Full:
                    pass
        # periodic GC
        now = time.monotonic()
        if now - last_gc > 0.1:
            reasm.gc()
            last_gc = now

def format_fps_latency(fps, latency_ms):
    return f"FPS: {fps:5.1f}  Latency: {latency_ms:6.1f} ms"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bind', type=str, default='0.0.0.0')
    ap.add_argument('--port', type=int, default=5005)
    ap.add_argument('--no-display', action='store_true', help='Disable display (receive only)')
    ap.add_argument('--side_by_side', action='store_true', help='Show RGB and Depth side-by-side in one window')
    args = ap.parse_args()

    # Two queues for stream 0 (RGB) and 1 (Depth)
    q_rgb = queue.Queue(maxsize=16)
    q_depth = queue.Queue(maxsize=16)
    out_queues = {0: q_rgb, 1: q_depth}

    rx_thread = threading.Thread(target=run_receiver, args=(args.bind, args.port, out_queues), daemon=True)
    rx_thread.start()

    # Stats
    last_rgb_time = time.time()
    last_depth_time = time.time()
    rgb_count = 0
    depth_count = 0
    rgb_fps = 0.0
    depth_fps = 0.0
    rgb_latency_ms = 0.0
    depth_latency_ms = 0.0

    last_fid_rgb = -1
    last_fid_depth = -1

    rgb_img = None
    depth_img = None

    try:
        while True:
            # Non-blocking get from queues
            try:
                fid, ts_us, frame_bytes = q_rgb.get(timeout=0.01)
                # Decode
                rgb_img = jpeg_decode(frame_bytes)
                now_us = time.time_ns() // 1_000
                rgb_latency_ms = (now_us - ts_us) / 1000.0
                rgb_count += 1
                last_fid_rgb = fid
            except queue.Empty:
                pass
            except Exception as e:
                # decoding error; skip
                pass

            try:
                fid, ts_us, frame_bytes = q_depth.get(timeout=0.0)
                depth_img = jpeg_decode(frame_bytes)
                now_us = time.time_ns() // 1_000
                depth_latency_ms = (now_us - ts_us) / 1000.0
                depth_count += 1
                last_fid_depth = fid
            except queue.Empty:
                pass
            except Exception as e:
                pass

            # Update FPS every second
            now = time.time()
            if now - last_rgb_time >= 1.0:
                rgb_fps = rgb_count / (now - last_rgb_time)
                rgb_count = 0
                last_rgb_time = now
            if now - last_depth_time >= 1.0:
                depth_fps = depth_count / (now - last_depth_time)
                depth_count = 0
                last_depth_time = now

            if not args.no_display:
                if args.side_by_side:
                    if rgb_img is not None and depth_img is not None:
                        # Match heights
                        h1, w1 = rgb_img.shape[:2]
                        h2, w2 = depth_img.shape[:2]
                        if h1 != h2:
                            scale = h1 / h2
                            depth_resized = cv2.resize(depth_img, (int(w2 * scale), h1))
                        else:
                            depth_resized = depth_img
                        combined = np.hstack([rgb_img, depth_resized])
                        cv2.putText(combined, format_fps_latency(rgb_fps, rgb_latency_ms), (12, 32),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
                        cv2.putText(combined, format_fps_latency(depth_fps, depth_latency_ms), (combined.shape[1]//2 + 12, 32),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)
                        cv2.imshow("Server RGB | Depth", combined)
                else:
                    if rgb_img is not None:
                        rgb_disp = rgb_img.copy()
                        cv2.putText(rgb_disp, format_fps_latency(rgb_fps, rgb_latency_ms), (12, 32),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
                        cv2.imshow("Server RGB", rgb_disp)
                    if depth_img is not None:
                        depth_disp = depth_img.copy()
                        cv2.putText(depth_disp, format_fps_latency(depth_fps, depth_latency_ms), (12, 32),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)
                        cv2.imshow("Server Depth", depth_disp)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
