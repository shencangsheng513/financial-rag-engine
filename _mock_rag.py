import http.server, socketserver, os

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = 5699

MIME = {
    'html': 'text/html; charset=utf-8',
    'js': 'application/javascript; charset=utf-8',
    'svg': 'image/svg+xml',
    'css': 'text/css; charset=utf-8',
    'json': 'application/json; charset=utf-8',
    'png': 'image/png',
    'ico': 'image/x-icon',
}

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def do_GET(self):
        p = self.path.split('?')[0]
        if p == '/' or p == '/index.html':
            self._serve(os.path.join(ROOT, 'templates', 'index.html'))
            return
        if p.startswith('/static/'):
            fp = os.path.join(ROOT, p.lstrip('/'))
            if os.path.isfile(fp):
                self._serve(fp)
                return
        self.send_error(404, 'not found')
    def _serve(self, fp):
        ext = os.path.splitext(fp)[1].lower().lstrip('.')
        ctype = MIME.get(ext, 'application/octet-stream')
        with open(fp, 'rb') as f:
            data = f.read()
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Disposition', 'inline')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, fmt, *a):
        print("[REQ]", fmt % a, flush=True)

with socketserver.TCPServer(("127.0.0.1", PORT), H) as httpd:
    print(f"mock rag server on {PORT}")
    httpd.serve_forever()
