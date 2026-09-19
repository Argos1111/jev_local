"""Image transport, validation and isolation from text prefix snapshots."""
import base64
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from image_input import load_image, validate_images
from jev_local import JevLocal
from systemone import SystemOne, ValidationError
from tests.test_jev_local import Fake
from tests.test_systemone import FakeBackend

PNG = b'\x89PNG\r\n\x1a\nfixture'
URL = 'data:image/png;base64,' + base64.b64encode(PNG).decode()


class ImageTests(unittest.TestCase):
    def test_validation_and_client_file_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'image.bin'
            path.write_bytes(PNG)
            self.assertEqual(load_image(path), URL)
        self.assertEqual(validate_images([URL]), [URL])
        for images in [None, URL, [URL]*5, ['https://example.com/image.png'], ['/etc/passwd'],
                       ['data:image/png;base64,%%%'], ['data:image/jpeg;base64,'+URL.split(',')[1]],
                       ['data:image/png;base64,'], ['data:image/png;base64,'+'A'*(6*1024*1024)]]:
            with self.subTest(images=str(images)[:80]), self.assertRaises(ValueError):
                validate_images(images)

    def test_native_image_prompt_and_logprob_retry(self):
        api = Fake()
        api.media_marker = '<server-marker>'
        result = api.score('same state', {'question':'Color?', 'choices':['red','blue']}, True, images=[URL])
        self.assertEqual(result['value'], 'blue')
        self.assertEqual(len(api.requests), 2)
        for request in api.requests:
            self.assertFalse(request['cache_prompt'])
            self.assertEqual(request['prompt']['multimodal_data'], [URL.split(',')[1]])
            self.assertIn('<server-marker>', request['prompt']['prompt_string'])
            self.assertNotIn(URL.split(',')[1], request['prompt']['prompt_string'])

    def test_text_only_backend_rejects_images(self):
        from io import BytesIO
        with patch('urllib.request.urlopen', return_value=BytesIO(json.dumps({'modalities':{'vision':False}}).encode())):
            with self.assertRaisesRegex(RuntimeError, 'no vision support'):
                JevLocal().prepare_images([URL])

    def test_images_lease_slots_without_snapshotting(self):
        class Backend(FakeBackend):
            def prepare_images(self, images): self.images = images
            def score(self, state, spec, **kwargs):
                assert kwargs == {'images':[URL], 'slot':3}
                return super().score(state, spec)
        class Cache:
            leases = 0
            @contextmanager
            def lease(self):
                self.leases += 1
                yield 3
            def run(self, *args): raise AssertionError('Text snapshot used for image')
            def close(self): pass
        backend, cache = Backend(), Cache()
        engine = SystemOne(backend_factory=lambda:backend, state_cache=False)
        engine.cache = cache
        self.addCleanup(engine.close)
        payload = {'model':'jev-latest','state':'Image','images':[URL], 'questions':{
            'a':{'type':'choice','criteria':{'red':None,'blue':None}},
            'b':{'type':'noul','instructions':'Is it red?'}}}
        diagnostics = {}
        result = engine.evaluate(payload, diagnostics=diagnostics)
        self.assertEqual(cache.leases, 2)
        self.assertEqual(diagnostics['mode'], 'off-images')
        self.assertEqual(result['usage']['output_tokens'], 2)
        payload['images'] = ['file:///etc/passwd']
        with self.assertRaises(ValidationError): engine.evaluate(payload)
        self.assertEqual(cache.leases, 2)
