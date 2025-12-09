import http.server
import socketserver
import os
import webbrowser

PORT = 8000
DIRECTORY = "quantum_cf_pulse/viz"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def main():
    # Ensure we are in the right directory context or change to project root
    # The DIRECTORY variable handles the relative path from project root
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving visualization at http://localhost:{PORT}")
        print("Press Ctrl+C to stop.")
        # Open browser automatically
        webbrowser.open(f"http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()
