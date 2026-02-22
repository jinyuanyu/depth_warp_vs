# depth_warp_vs/udp_stream_client.py
import argparse
import socket
import struct
import time
import threading
import cv2
import numpy as np

try:
    from turbojpeg import TurboJPEG, TJPF_BGR, TJSAMP_420
    _jpeg = TurboJPEG()
    def jpeg_encode(img, quality):
        return _jpeg.encode(img, quality=quality, pixel_format=TJPF_BGR, jpeg_subsample=TJSAMP_420)
except Exception:
    _jpeg = None
    def jpeg_encode(img, quality):
        ret, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ret:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()

MAGIC = b'IM'
VERSION = 1
# Header: magic(2s), version(B), stream(B), frame_id(I), timestamp_us(Q), total_chunks(H), chunk_id(H), payload_len(H)
HEADER_STRUCT = struct.Struct('!2s B B I Q H H H')
HEADER_SIZE = HEADER_STRUCT.size

def chunk_and_send(sock, addr, stream_id, frame_id, timestamp_us, data, mtu):
    max_payload = mtu - HEADER_SIZE
    total_chunks = (len(data) + max_payload - 1) // max_payload
    view = memoryview(data)
    for chunk_id in range(total_chunks):
        start = chunk_id * max_payload
        end = min(start + max_payload, len(data))
        payload = view[start:end]
        header = HEADER_STRUCT.pack(
            MAGIC, VERSION, stream_id, frame_id, timestamp_us,
            total_chunks, chunk_id, len(payload)
        )
        sock.sendto(header + payload, addr)

def cap_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps > 1200:
        return None
    return fps

def reopen_if_video_end(cap, is_video):
    if not is_video:
        return True
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if total > 0 and pos >= total - 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server-ip', type=str, required=True, help='Server IP (LAN IP or 127.0.0.1)')
    ap.add_argument('--port', type=int, default=5005)
    ap.add_argument('--camera-index', type=int, default=0)
    ap.add_argument('--video', type=str, default="/media/a1/16THDD/Zhan/left_eye.mp4", help='RGB video path for testing')
    ap.add_argument('--depth-video', type=str, default="/media/a1/16THDD/Zhan/111_depth.mp4", help='Depth video path for testing')
    ap.add_argument('--width', type=int, default=0)
    ap.add_argument('--height', type=int, default=0)
    ap.add_argument('--fps', type=float, default=0.0, help='Force send FPS (ignored: unlimited mode)')
    ap.add_argument('--jpeg_quality', type=int, default=85)
    ap.add_argument('--mtu', type=int, default=1400)
    ap.add_argument('--show', action='store_true', help='Show local preview')
    args = ap.parse_args()

    addr = (args.server_ip, args.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Increase send buffer and set low-latency DSCP if possible
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    except Exception:
        pass
    try:
        # Expedited forwarding DSCP (46 -> 0x2E). Some stacks use IP_TOS.
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0xB8)  # EF
    except Exception:
        pass

    # Open RGB source
    if args.video:
        rgb_cap = cv2.VideoCapture(args.video)
        is_video_rgb = True
    else:
        rgb_cap = cv2.VideoCapture(args.camera_index)
        is_video_rgb = False

    if not rgb_cap.isOpened():
        raise RuntimeError("Failed to open RGB source")

    # Set resolution if provided (for camera)
    if not args.video and (args.width > 0 and args.height > 0):
        rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Open depth source (optional); if not provided, we will copy RGB frame
    depth_cap = None
    is_video_depth = False
    if args.depth_video:
        depth_cap = cv2.VideoCapture(args.depth_video)
        if not depth_cap.isOpened():
            print("Warn: failed to open depth video, will use RGB-copy as depth.")
            depth_cap = None
        else:
            is_video_depth = True

    print(f"Sending to {addr}, unlimited fps, mtu={args.mtu}, quality={args.jpeg_quality}")

    frame_id = 0

    try:
        while True:
            ok_rgb, rgb = rgb_cap.read()
            if not ok_rgb:
                if not reopen_if_video_end(rgb_cap, is_video_rgb):
                    break
                continue

            # Prepare depth frame
            if depth_cap is not None:
                ok_d, depth = depth_cap.read()
                if not ok_d:
                    if not reopen_if_video_end(depth_cap, is_video_depth):
                        depth = rgb.copy()
                    else:
                        continue
            else:
                depth = rgb.copy()

            # Encode
            ts_us = time.time_ns() // 1_000
            rgb_jpg = jpeg_encode(rgb, args.jpeg_quality)
            depth_jpg = jpeg_encode(depth, args.jpeg_quality)

            # Send RGB (stream 0) and Depth (stream 1)
            chunk_and_send(sock, addr, 0, frame_id, ts_us, rgb_jpg, args.mtu)
            chunk_and_send(sock, addr, 1, frame_id, ts_us, depth_jpg, args.mtu)

            if args.show:
                preview = rgb.copy()
                cv2.putText(preview, f"Sent frame {frame_id}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow("Client Preview (RGB)", preview)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_id += 1

            # No pacing: send as fast as possible

    except KeyboardInterrupt:
        pass
    finally:
        rgb_cap.release()
        if depth_cap is not None:
            depth_cap.release()
        sock.close()
        if args.show:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# python -m depth_warp_vs.udp_stream_client --server-ip 192.168.31.91 --port 5005
