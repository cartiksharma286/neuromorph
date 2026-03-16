"""Quick API test runner"""
import sys, traceback, json
sys.path.insert(0, '.')

from app_enhanced import app

client = app.test_client()

tests = [
    ('/', 'GET', None),
    ('/api/simulate', 'POST', {'resolution':64,'sequence':'SE','tr':2000,'te':100}),
    ('/api/quantum_coils/list', 'GET', None),
    ('/api/head_coil_50/specs', 'GET', None),
    ('/api/adaptive_sequence/generate', 'POST', {'type':'adaptive_se'}),
    ('/api/schematics/generate', 'GET', None),
    ('/api/neuro_pulse_ca/generate', 'POST', {}),
    ('/api/neurovasculature/render', 'POST', {'enable_50_turn': True}),
    ('/api/thermometry_stream', 'GET', None),
    ('/api/signal_reconstruction/coil_geometry', 'POST', {'coil_types':['standard']}),
    ('/api/robotics/optimize_coils', 'POST', {'target':'Circle of Willis'}),
    ('/api/render_cortical', 'POST', {}),
]

for path, method, body in tests:
    try:
        if method == 'GET':
            r = client.get(path)
        else:
            r = client.post(path, json=body)
        
        try:
            data = json.loads(r.data.decode())
            ok = data.get('success', False)
            err = data.get('error', '') if not ok else ''
            print(f'[{r.status_code}] {method} {path}: success={ok} {err}')
        except Exception:
            print(f'[{r.status_code}] {method} {path}: non-JSON response (len={len(r.data)})')
    except Exception as e:
        print(f'EXCEPTION {method} {path}: {e}')
        traceback.print_exc()
