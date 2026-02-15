
import http.server
import socketserver
import os
import json
import glob

PORT = 8080
# Assuming this script is in backend/
MATCH_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MATCH_DIR) # Go up one level to project root

class IGSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # API: Latest Session
        if self.path == '/api/latest-session':
            try:
                sessions_dir = os.path.join(PROJECT_ROOT, 'backend', 'data', 'sessions')
                # Find all session files (exclude igs_plan if they are mixed, but they are named session_*)
                list_of_files = glob.glob(os.path.join(sessions_dir, 'session_*.json'))
                
                if not list_of_files:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'{"error": "No sessions found"}')
                    return
                    
                # Sort by localized timestamp in filename or creation time
                # Using creation time for simplicity
                latest_file = max(list_of_files, key=os.path.getctime)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                with open(latest_file, 'rb') as f:
                    self.wfile.write(f.read())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())
            return

        # Routing for Frontend
        if self.path == '/' or self.path == '/index.html':
            self.path = 'frontend/index.html'
        elif self.path == '/style.css':
            self.path = 'frontend/style.css'
        elif self.path == '/app.js':
            self.path = 'frontend/app.js'
        
        # Routing for Data (Frontend expects /data/...)
        elif self.path.startswith('/data/'):
            # Map /data/x to backend/data/x
            # The SimpleHTTPRequestHandler serves relative to CWD
            self.path = 'backend' + self.path
            
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

print(f"Starting IGS Platform Server at http://localhost:{PORT}")
print(f"Serving from {PROJECT_ROOT}")

# Switch to project root so 'frontend/' and 'backend/' paths work
os.chdir(PROJECT_ROOT)

with socketserver.TCPServer(("", PORT), IGSRequestHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
