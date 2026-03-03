#!/usr/bin/env python3
"""
Keep-alive script for Render deployments.
Pings your API at regular intervals to prevent it from spinning down.
"""

import os
import time
import logging
import urllib.request
import urllib.error
from datetime import datetime

# --- Configuration ---
PING_URL      = "https://talus-machine-learning.onrender.com/health"
PING_INTERVAL = 420 # 7 minutes
PING_TIMEOUT  = 10

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("keep_alive")


def ping(url: str, timeout: int) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            log.info(f"Pinged {url} → HTTP {resp.status}")
            return True
    except urllib.error.HTTPError as e:
        log.warning(f"HTTP error {e.code} from {url}")
        return False
    except urllib.error.URLError as e:
        log.error(f"Failed to reach {url}: {e.reason}")
        return False
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        return False


def main():
    log.info("Keep-alive started")
    log.info(f"    URL      : {PING_URL}")
    log.info(f"    Interval : {PING_INTERVAL}s ({PING_INTERVAL // 60}m {PING_INTERVAL % 60}s)")
    log.info(f"    Timeout  : {PING_TIMEOUT}s")

    consecutive_failures = 0

    while True:
        success = ping(PING_URL, PING_TIMEOUT)

        if success:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                log.critical(f"{consecutive_failures} consecutive failures — check your service!")

        time.sleep(PING_INTERVAL)


if __name__ == "__main__":
    main()