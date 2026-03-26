import base64
import argparse
import hashlib
import hmac
import http.client
import json
import logging
import msvcrt
import os
import pathlib
import random
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote, urlencode, urlsplit


AUTH_HOST = "172.19.0.5"
DNS_HOST = "10.1.20.30"
CHECK_INTERVAL_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 8
LOGIN_N = 200
LOGIN_TYPE = 1
LOGIN_ENC = "srun_bx1"
CUSTOM_B64_ALPHA = "LVoJPiCN2R8G90yg+hmFHuacZ1OWMnrsSTXkYpUq/3dlbfKwv6xztjI7DeBE45QA"

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR / "workspace"

LOG_DIR = WORKSPACE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "campus-net-guard.log"
LOCK_FILE = LOG_DIR / "campus-net-guard.lock"
CONFIG: dict = {}


def resolve_config_file() -> pathlib.Path:
    candidates = [
        SCRIPT_DIR / "campus-net-guard.json",
        WORKSPACE_DIR / "campus-net-guard.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_config() -> dict:
    config_file = resolve_config_file()
    with config_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    required = ["account", "password", "domain", "ac_id"]
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ValueError(f"missing required config keys in {config_file}: {', '.join(missing)}")
    return data


@dataclass
class CampusRoute:
    gateway: str
    interface_ip: str


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def acquire_single_instance_lock():
    handle = open(LOCK_FILE, "a+")
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        logging.error("another campus net guard instance is already running")
        sys.exit(1)
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )


def is_private_ip(ip: str) -> bool:
    return (
        ip.startswith("10.")
        or ip.startswith("192.168.")
        or re.match(r"^172\.(1[6-9]|2\d|3[0-1])\.", ip) is not None
    )


def discover_campus_route() -> Optional[CampusRoute]:
    result = run_command(["route", "print", "-4"])
    if result.returncode != 0:
        logging.warning("route print failed: %s", result.stderr.strip())
        return None

    lines = result.stdout.splitlines()
    candidates: list[tuple[int, CampusRoute]] = []
    for raw_line in lines:
        cols = raw_line.split()
        if len(cols) != 5:
            continue
        dest, mask, gateway, interface_ip, metric = cols
        if dest != "0.0.0.0" or mask != "0.0.0.0":
            continue
        if gateway.startswith("198.18.") or interface_ip.startswith("198.18."):
            continue
        if not is_private_ip(gateway) or not is_private_ip(interface_ip):
            continue
        try:
            metric_value = int(metric)
        except ValueError:
            continue
        candidates.append((metric_value, CampusRoute(gateway=gateway, interface_ip=interface_ip)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def route_exists(target: str) -> bool:
    result = run_command(["route", "print", target])
    return target in result.stdout


def ensure_host_route(target: str, route: CampusRoute) -> None:
    if route_exists(target):
        return
    result = run_command(
        ["route", "add", target, "mask", "255.255.255.255", route.gateway, "metric", "5"]
    )
    if result.returncode == 0:
        logging.info("added host route %s via %s", target, route.gateway)


def direct_http_get(url: str, source_ip: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname
    if not host:
        raise ValueError(f"invalid URL: {url}")
    port = parts.port or 80
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"

    conn = http.client.HTTPConnection(
        host=host,
        port=port,
        timeout=REQUEST_TIMEOUT_SECONDS,
        source_address=(source_ip, 0),
    )
    try:
        conn.request(
            "GET",
            path,
            headers={
                "Host": host if parts.port is None else f"{host}:{port}",
                "User-Agent": "campus-net-guard/1.0",
                "Connection": "close",
            },
        )
        response = conn.getresponse()
        payload = response.read()
        return payload.decode("utf-8", "ignore")
    finally:
        conn.close()


def parse_jsonp(payload: str) -> dict:
    match = re.search(r"\((\{.*\})\)\s*$", payload.strip(), re.S)
    if not match:
        raise ValueError(f"unexpected JSONP payload: {payload[:120]}")
    return json.loads(match.group(1))


def sencode(msg: str, include_length: bool) -> list[int]:
    length = len(msg)
    result: list[int] = []
    for index in range(0, length, 4):
        value = 0
        for shift in range(4):
            pos = index + shift
            if pos < length:
                value |= ord(msg[pos]) << (shift * 8)
        result.append(value & 0xFFFFFFFF)
    if include_length:
        result.append(length)
    return result


def lencode(values: list[int], include_length: bool) -> str:
    chars: list[str] = []
    for value in values:
        chars.append(chr(value & 0xFF))
        chars.append(chr((value >> 8) & 0xFF))
        chars.append(chr((value >> 16) & 0xFF))
        chars.append(chr((value >> 24) & 0xFF))
    data = "".join(chars)
    if include_length:
        length = values[-1]
        return data[:length]
    return data


def xencode(msg: str, key: str) -> str:
    if not msg:
        return ""
    v = sencode(msg, True)
    k = sencode(key, False)
    while len(k) < 4:
        k.append(0)
    n = len(v) - 1
    z = v[n]
    c = 0x86014019 | 0x183639A0
    q = int(6 + 52 / (n + 1))
    d = 0
    while q > 0:
        d = (d + c) & 0xFFFFFFFF
        e = (d >> 2) & 3
        for p in range(n):
            y = v[p + 1]
            mx = (z >> 5) ^ (y << 2)
            mx = (mx + ((y >> 3) ^ (z << 4) ^ (d ^ y))) & 0xFFFFFFFF
            mx = (mx + (k[(p & 3) ^ e] ^ z)) & 0xFFFFFFFF
            v[p] = (v[p] + mx) & 0xFFFFFFFF
            z = v[p]
        y = v[0]
        mx = (z >> 5) ^ (y << 2)
        mx = (mx + ((y >> 3) ^ (z << 4) ^ (d ^ y))) & 0xFFFFFFFF
        mx = (mx + (k[(n & 3) ^ e] ^ z)) & 0xFFFFFFFF
        v[n] = (v[n] + mx) & 0xFFFFFFFF
        z = v[n]
        q -= 1
    return lencode(v, False)


def custom_b64_encode(data: str) -> str:
    std = base64.b64encode(data.encode("latin1")).decode("ascii")
    standard_alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    translation = str.maketrans(standard_alpha, CUSTOM_B64_ALPHA)
    return std.translate(translation)


def build_callback() -> str:
    now = int(time.time() * 1000)
    return f"jQuery{now}_{random.randint(100, 999)}"


def build_username() -> str:
    return f"{CONFIG['account']}{CONFIG['domain']}"


def get_online_info(source_ip: str) -> Optional[str]:
    payload = direct_http_get(f"http://{AUTH_HOST}/cgi-bin/rad_user_info", source_ip)
    text = payload.strip()
    if not text or "not_online" in text.lower():
        return None
    parts = text.split(",")
    if not parts or parts[0] != CONFIG["account"]:
        return None
    return text


def get_challenge(source_ip: str) -> dict:
    callback = build_callback()
    params = {
        "callback": callback,
        "username": build_username(),
        "ip": source_ip,
        "_": int(time.time() * 1000),
    }
    payload = direct_http_get(f"http://{AUTH_HOST}/cgi-bin/get_challenge?{urlencode(params)}", source_ip)
    return parse_jsonp(payload)


def login(source_ip: str) -> bool:
    challenge = get_challenge(source_ip)
    token = challenge["challenge"]
    ip = challenge.get("client_ip") or challenge.get("online_ip") or source_ip
    username = build_username()
    hmd5 = hmac.new(
        token.encode("utf-8"),
        CONFIG["password"].encode("utf-8"),
        hashlib.md5,
    ).hexdigest()
    info_obj = {
        "username": username,
        "password": CONFIG["password"],
        "ip": ip,
        "acid": CONFIG["ac_id"],
        "enc_ver": LOGIN_ENC,
    }
    info_json = json.dumps(info_obj, separators=(",", ":"), ensure_ascii=False)
    encoded_info = "{SRBX1}" + custom_b64_encode(xencode(info_json, token))
    checksum = (
        token + username + token + hmd5 + token + CONFIG["ac_id"] + token + ip + token
        + str(LOGIN_N) + token + str(LOGIN_TYPE) + token + encoded_info
    )
    callback = build_callback()
    params = {
        "callback": callback,
        "action": "login",
        "username": username,
        "password": "{MD5}" + hmd5,
        "os": "Windows",
        "name": "Windows",
        "double_stack": "0",
        "chksum": hashlib.sha1(checksum.encode("utf-8")).hexdigest(),
        "info": encoded_info,
        "ac_id": CONFIG["ac_id"],
        "ip": ip,
        "n": str(LOGIN_N),
        "type": str(LOGIN_TYPE),
    }
    query = "&".join(f"{key}={quote(value, safe='{}@:_')}" for key, value in params.items())
    payload = direct_http_get(f"http://{AUTH_HOST}/cgi-bin/srun_portal?{query}", source_ip)
    result = parse_jsonp(payload)
    error_msg = str(result.get("error_msg", ""))
    suc_msg = str(result.get("suc_msg", ""))
    if result.get("error") == "ok" or result.get("res") == "ok":
        logging.info("campus login accepted: %s %s", suc_msg, error_msg)
        return True
    if "ip_already_online_error" in error_msg or "ip_already_online_error" in suc_msg:
        logging.info("campus login reports already online")
        return True
    logging.warning("campus login failed: %s", result)
    return False


def run_once(last_status: Optional[str] = None) -> str:
    route = discover_campus_route()
    if not route:
        if last_status != "no_route":
            logging.warning("no campus route discovered")
        return "no_route"

    ensure_host_route(AUTH_HOST, route)
    ensure_host_route(DNS_HOST, route)

    online = get_online_info(route.interface_ip)
    if online:
        if last_status != "online":
            logging.info("campus online via %s", route.interface_ip)
        return "online"

    if last_status != "offline":
        logging.warning("campus offline detected; attempting login via %s", route.interface_ip)
    if login(route.interface_ip):
        time.sleep(5)
        verify = get_online_info(route.interface_ip)
        if verify:
            logging.info("campus login verified")
            return "online"
        logging.warning("campus login returned success but verification is still offline")
    return "offline"


def loop() -> None:
    logging.info("campus net guard started")
    last_status: Optional[str] = None
    while True:
        try:
            last_status = run_once(last_status)
        except (OSError, socket.error, TimeoutError, ValueError) as exc:
            logging.warning("loop error: %s", exc)
        except Exception as exc:
            logging.exception("unexpected error: %s", exc)
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    setup_logging()
    CONFIG = load_config()
    if args.once:
        run_once()
    else:
        _lock_handle = acquire_single_instance_lock()
        loop()
