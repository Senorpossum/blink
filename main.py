"""
BlinkFlow Entrypoint & CLI Launcher
Runs the local vision server and opens the Control Center in your browser.
"""

import argparse
import sys
import threading
import time
import webbrowser
import uvicorn


BANNER = r"""
  ____  _ _       _     _____ _                 
 |  _ \| (_)_ __ | | __|  ___| | _____      __  
 | |_) | | | '_ \| |/ /| |_  | |/ _ \ \ /\ / /  
 |  _ <| | | | | |   < |  _| | | (_) \ V  V /   
 |_| \_\_|_|_| |_|_|\_\|_|   |_|\___/ \_/\_/    
  Real-time Eye Blink & Gesture Shortcut Controller
"""


def open_browser(url: str, delay: float = 1.5):
    time.sleep(delay)
    print(f"\n[BlinkFlow] Opening dashboard at {url} ...")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[BlinkFlow] Note: Could not auto-open browser ({e}). Visit {url} manually.")


def main():
    parser = argparse.ArgumentParser(description="BlinkFlow: Camera Blink & Shortcut Controller")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the web dashboard")
    args = parser.parse_args()

    print(BANNER)
    dashboard_url = f"http://{args.host}:{args.port}"
    print(f"[BlinkFlow] Starting local server on {dashboard_url}")
    print("[BlinkFlow] Press Ctrl+C to terminate.\n")

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(dashboard_url,), daemon=True).start()

    try:
        uvicorn.run("server.app:app", host=args.host, port=args.port, log_level="info")
    except KeyboardInterrupt:
        print("\n[BlinkFlow] Shutting down gracefully. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
