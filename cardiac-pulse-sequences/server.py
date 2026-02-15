import http.server
import socketserver
import json
import os
import sys
import subprocess
from quantum_optimizer import QuantumParallelOptimizer

PORT = 8002
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Initialize global optimizer
optimizer = QuantumParallelOptimizer()

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == '/api/quantum-optimize':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                acceleration_factor = float(data.get('accelerationFactor', 2.0))
                coil_elements = int(data.get('coilElements', 18))
                
                print(f"Optimizing for R={acceleration_factor}, Coils={coil_elements}")
                
                result = optimizer.optimize_sampling_pattern(acceleration_factor, coil_elements)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
                
            except Exception as e:
                print(f"Error processing request: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
        
        elif self.path == '/api/write-seq':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                python_code = data.get('code')
                
                if not python_code:
                    raise ValueError("No code provided")

                print("Executing PyPulseq code in-memory...")
                
                # Execute the code using exec() to avoid subprocess overhead
                import io
                from contextlib import redirect_stdout
                
                # Capture stdout
                f = io.StringIO()
                try:
                    with redirect_stdout(f):
                        # Create a restricted globals dictionary but ensure necessary builtins are available
                        # imports inside the code string (like import pypulseq) will work in local scope
                        exec_globals = {"__builtins__": __builtins__}
                        exec(python_code, exec_globals)
                    
                    output = f.getvalue()
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'status': 'success', 
                        'message': 'Sequence file written successfully!',
                        'output': output
                    }).encode('utf-8'))
                    
                except Exception as exec_error:
                    print(f"Execution failed: {exec_error}")
                    raise Exception(f"Execution failed: {exec_error}")

            except Exception as e:
                print(f"Error processing write-seq request: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving at http://localhost:{PORT}")
            print("Quantum Optimization API active at /api/quantum-optimize")
            httpd.serve_forever()
    except OSError as e:
        print(f"Error: {e}")
        print(f"Port {PORT} might be in active use. Try a different port.")

if __name__ == "__main__":
    os.chdir(DIRECTORY)
    run_server()
