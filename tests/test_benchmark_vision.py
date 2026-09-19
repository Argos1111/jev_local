"""The published benchmark must run without a local-only script or image path."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import benchmark_vision


class Response(io.BytesIO):
    headers = {'X-Jev-Local-State-Cache':'off-images'}


class VisionBenchmarkTests(unittest.TestCase):
    def test_warmup_excluded_and_payload_sent_with_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pic, source, output = root/'test.png', root/'request.json', root/'out/report.json'
            pic.write_bytes(b'\x89PNG\r\n\x1a\nfixture')
            payload = {'model':'jev-latest','state':'read it','questions':{'q':{'type':'choice','criteria':{'red':None,'blue':None}}}}
            source.write_text(json.dumps(payload))
            response = {'model':'test-vl','answers':{'q':{'type':'choice','choice':'red','probabilities':{'red':.9,'blue':.1}}},'usage':{'input_tokens':100,'output_tokens':1}}
            requests = []
            def open_request(req, timeout):
                requests.append(json.loads(req.data))
                return Response(json.dumps(response).encode())
            argv = ['benchmark_vision','--input',str(source),'--image',str(pic),'--output',str(output),'--rounds','2']
            with patch('sys.argv',argv), patch.object(benchmark_vision.urllib.request,'urlopen',side_effect=open_request), patch.object(benchmark_vision.time,'perf_counter',side_effect=[0,1,2,2.1,3,3.3]), redirect_stdout(io.StringIO()):
                benchmark_vision.main()
            data = json.loads(output.read_text())
            self.assertEqual(len(requests),3)
            self.assertTrue(all(r['images'][0].startswith('data:image/png;base64,') for r in requests))
            self.assertEqual(data['request_without_images'],payload)
            self.assertNotIn('data:image',output.read_text())
            self.assertEqual(data['warmup']['elapsed_ms'],1000)
            self.assertEqual(len(data['runs']),2)
            self.assertAlmostEqual(data['summary_ms']['median'],200)

    def test_missing_input_fails_before_http(self):
        with patch('sys.argv',['benchmark_vision','--image','/nonexistent/image.png']), patch.object(benchmark_vision.urllib.request,'urlopen') as request, redirect_stdout(io.StringIO()), patch('sys.stderr',io.StringIO()):
            with self.assertRaises(SystemExit): benchmark_vision.main()
            request.assert_not_called()
