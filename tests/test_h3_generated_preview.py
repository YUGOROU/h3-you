import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from urllib.request import urlopen
from urllib.error import HTTPError
from research.serve_h3_generated_preview import handler,HTML


class PreviewTests(unittest.TestCase):
    def test_manual_only(self):
        self.assertNotIn(b'autoplay',HTML);self.assertNotIn(b'.play(',HTML);self.assertNotIn(b'navigator',HTML)

    def test_only_verified_files_and_no_capture_permissions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);body=b'generated';(root/'segment-000.mp4').write_bytes(body)
            (root/'receive-receipt.json').write_text(json.dumps({'segments':{'segment-000.mp4':{'sha256':hashlib.sha256(body).hexdigest(),'pts_full_decode_verified':True}}}))
            server=ThreadingHTTPServer(('127.0.0.1',0),handler(root));thread=Thread(target=server.serve_forever);thread.start()
            try:
                base=f'http://127.0.0.1:{server.server_port}'
                with urlopen(base+'/segment-000.mp4') as response:
                    self.assertEqual(response.read(),body)
                    self.assertIn('camera=()',response.headers['Permissions-Policy'])
                for path in ('/reference.jpeg','/receive-receipt.json','/segment-001.mp4','/../secret'):
                    with self.assertRaises(HTTPError):urlopen(base+path)
                (root/'segment-000.mp4').write_bytes(b'tampered')
                with self.assertRaises(HTTPError) as error:urlopen(base+'/segment-000.mp4')
                self.assertEqual(error.exception.code,409)
            finally:server.shutdown();thread.join();server.server_close()

if __name__=='__main__':unittest.main()
