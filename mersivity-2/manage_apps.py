#!/usr/bin/env python3
"""
Mersivity Dual-App Lifecycle & Streamlined Management Engine
Manages:
  1. 3D Neuro-Registration Suite (app_registration.py -> Port 5050)
  2. BCI & TMS Neuromodulation Suite (app_bci_tms.py -> Port 5055)
"""

import os
import sys
import time
import signal
import subprocess
import urllib.request
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(SCRIPT_DIR, '.run')
LOG_DIR = os.path.join(SCRIPT_DIR, '.logs')
VENV_PYTHON = os.path.join(SCRIPT_DIR, '.venv', 'bin', 'python')
PYTHON_EXE = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

REG_PORT = int(os.environ.get('REGISTRATION_PORT', 5050))
BCI_PORT = int(os.environ.get('BCI_TMS_PORT', 5055))

APPS = {
    'registration': {
        'name': '3D Neuro-Registration Suite',
        'script': os.path.join(SCRIPT_DIR, 'app_registration.py'),
        'port': REG_PORT,
        'pid_file': os.path.join(RUN_DIR, 'registration.pid'),
        'log_file': os.path.join(LOG_DIR, 'registration.log'),
        'health_url': f'http://127.0.0.1:{REG_PORT}/api/health'
    },
    'bci_tms': {
        'name': 'BCI & TMS Neuromodulation Suite',
        'script': os.path.join(SCRIPT_DIR, 'app_bci_tms.py'),
        'port': BCI_PORT,
        'pid_file': os.path.join(RUN_DIR, 'bci_tms.pid'),
        'log_file': os.path.join(LOG_DIR, 'bci_tms.log'),
        'health_url': f'http://127.0.0.1:{BCI_PORT}/api/health'
    }
}

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def is_pid_running(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_running_pid(app_key):
    pid_file = APPS[app_key]['pid_file']
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            if is_pid_running(pid):
                return pid
            else:
                os.remove(pid_file)
        except Exception:
            pass
    return None


def check_health(app_key, timeout=2.0):
    url = APPS[app_key]['health_url']
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'MersivityManager/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                return True, data.get('status', 'healthy')
    except Exception as e:
        return False, str(e)
    return False, 'Unreachable'


def start_app(app_key, wait=True):
    app_info = APPS[app_key]
    pid = get_running_pid(app_key)
    if pid:
        print(f"  [✓] {app_info['name']} is already running (PID: {pid}, Port: {app_info['port']})")
        return pid

    log_path = app_info['log_file']
    log_file = open(log_path, 'a')

    env = os.environ.copy()
    if app_key == 'registration':
        env['PORT'] = str(app_info['port'])
        env['REGISTRATION_PORT'] = str(app_info['port'])
    else:
        env['PORT'] = str(app_info['port'])
        env['BCI_TMS_PORT'] = str(app_info['port'])

    proc = subprocess.Popen(
        [PYTHON_EXE, app_info['script']],
        cwd=SCRIPT_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True
    )

    with open(app_info['pid_file'], 'w') as f:
        f.write(str(proc.pid))

    print(f"  [➔] Starting {app_info['name']} (PID: {proc.pid}, Port: {app_info['port']})...")

    if wait:
        ready = False
        for _ in range(30):
            time.sleep(0.5)
            healthy, msg = check_health(app_key, timeout=1.0)
            if healthy:
                ready = True
                break
            if proc.poll() is not None:
                print(f"  [✗] Process exited prematurely with code {proc.returncode}. See {log_path}")
                return None
        if ready:
            print(f"  [✓] {app_info['name']} is healthy and listening on http://localhost:{app_info['port']}")
        else:
            print(f"  [!] {app_info['name']} process started (PID: {proc.pid}), still warming up...")

    return proc.pid


def stop_app(app_key):
    app_info = APPS[app_key]
    pid = get_running_pid(app_key)
    if not pid:
        print(f"  [-] {app_info['name']} is not running.")
        return True

    print(f"  [➔] Stopping {app_info['name']} (PID: {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.2)
            if not is_pid_running(pid):
                break
        if is_pid_running(pid):
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.2)
    except Exception as e:
        print(f"  [!] Error stopping PID {pid}: {e}")

    if os.path.exists(app_info['pid_file']):
        os.remove(app_info['pid_file'])
    print(f"  [✓] {app_info['name']} stopped.")
    return True


def status():
    print("\n" + "=" * 65)
    print("      MERSIVITY DUAL-APP MANAGEMENT STATUS DASHBOARD")
    print("=" * 65)
    
    for key, info in APPS.items():
        pid = get_running_pid(key)
        if pid:
            healthy, msg = check_health(key)
            health_str = f"\033[92mHEALTHY\033[0m ({msg})" if healthy else f"\033[93mINITIALIZING\033[0m ({msg})"
            print(f"• \033[1m{info['name']}\033[0m")
            print(f"    Status:   \033[92mRUNNING\033[0m")
            print(f"    PID:      {pid}")
            print(f"    URL:      http://localhost:{info['port']}")
            print(f"    Health:   {health_str}")
            print(f"    Log:      {info['log_file']}")
        else:
            print(f"• \033[1m{info['name']}\033[0m")
            print(f"    Status:   \033[91mSTOPPED\033[0m")
            print(f"    Port:     {info['port']}")
            print(f"    Script:   {info['script']}")
        print("-" * 65)
    print()


def start_all():
    print("\n" + "=" * 65)
    print("       LAUNCHING MERSIVITY STREAMLINED APP SERVICES")
    print("=" * 65)
    start_app('registration')
    start_app('bci_tms')
    status()


def stop_all():
    print("\n" + "=" * 65)
    print("       STOPPING ALL MERSIVITY APP SERVICES")
    print("=" * 65)
    stop_app('registration')
    stop_app('bci_tms')
    print("All services stopped.\n")


def restart_all():
    stop_all()
    time.sleep(1)
    start_all()


def show_logs(app_choice='all'):
    targets = [app_choice] if app_choice in APPS else list(APPS.keys())
    files = [APPS[k]['log_file'] for k in targets]
    print(f"Tailing logs for {', '.join(targets)}...")
    cmd = ['tail', '-n', '30', '-f'] + files
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nLog tailing stopped.")


def run_foreground():
    print("\n" + "=" * 65)
    print("       RUNNING MERSIVITY DUAL-APPS IN FOREGROUND")
    print("=" * 65)
    print("Press Ctrl+C to gracefully stop both servers.\n")
    
    procs = []
    
    def shutdown_handler(sig, frame):
        print("\n[!] Received interrupt signal. Shutting down apps gracefully...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            p.wait()
        print("[✓] Both applications terminated.")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    for key, info in APPS.items():
        env = os.environ.copy()
        env['PORT'] = str(info['port'])
        if key == 'registration':
            env['REGISTRATION_PORT'] = str(info['port'])
        else:
            env['BCI_TMS_PORT'] = str(info['port'])
            
        p = subprocess.Popen(
            [PYTHON_EXE, info['script']],
            cwd=SCRIPT_DIR,
            env=env
        )
        procs.append(p)
        print(f"[➔] Launched {info['name']} on http://localhost:{info['port']} (PID: {p.pid})")

    # Monitor
    try:
        while True:
            time.sleep(1)
            for i, p in enumerate(procs):
                ret = p.poll()
                if ret is not None:
                    print(f"[✗] An app process exited with code {ret}. Shutting down remaining...")
                    shutdown_handler(None, None)
    except KeyboardInterrupt:
        shutdown_handler(None, None)


def main():
    parser = argparse.ArgumentParser(description="Mersivity Dual-App Lifecycle Manager")
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status', 'run', 'logs'],
                        default='start', nargs='?', help="Action to execute")
    parser.add_argument('--app', choices=['registration', 'bci_tms', 'all'], default='all',
                        help="Target specific application")

    args = parser.parse_args()

    if args.action == 'start':
        if args.app == 'all':
            start_all()
        else:
            start_app(args.app)
            status()
    elif args.action == 'stop':
        if args.app == 'all':
            stop_all()
        else:
            stop_app(args.app)
    elif args.action == 'restart':
        if args.app == 'all':
            restart_all()
        else:
            stop_app(args.app)
            time.sleep(1)
            start_app(args.app)
            status()
    elif args.action == 'status':
        status()
    elif args.action == 'run':
        run_foreground()
    elif args.action == 'logs':
        show_logs(args.app)


if __name__ == '__main__':
    main()
