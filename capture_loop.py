import time
import os
from datetime import datetime

from capture_vrchat import find_vrchat_window
from capture_wgc import capture_window_wgc


def main(interval: float = 1.0, prefix: str = 'capture_loop'):
    out_dir = os.path.join(os.getcwd(), 'captures')
    os.makedirs(out_dir, exist_ok=True)

    print(f'Starting WGC capture loop: interval={interval}s, out_dir={out_dir}')
    while True:
        hwnd, title = find_vrchat_window()
        if not hwnd:
            time.sleep(interval)
            continue

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(out_dir, f'{prefix}_{ts}.png')
        try:
            capture_window_wgc('VRChat', out_path, timeout=5.0)
            print(f'saved {out_path}')
        except Exception as e:
            print(f'WGC capture failed: {e}')

        time.sleep(interval)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=float, default=1.0, help='Seconds between captures')
    parser.add_argument('--prefix', type=str, default='capture_loop', help='Filename prefix')
    args = parser.parse_args()
    main(interval=args.interval, prefix=args.prefix)
