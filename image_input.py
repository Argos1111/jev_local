"""Local image attachments; HTTP requests accept data URLs, never file paths."""
import base64
import binascii
from pathlib import Path

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGES = 4


def image_mime(data):
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    raise ValueError('Only PNG and JPEG images are supported')


def validate_images(images):
    if not isinstance(images, list) or len(images) > MAX_IMAGES:
        raise ValueError(f'images must be an array of at most {MAX_IMAGES} data URLs')
    for url in images:
        if not isinstance(url, str) or len(url) > 4 * ((MAX_IMAGE_BYTES + 2) // 3) + 32:
            raise ValueError('Each image must be a data URL of at most 4 MiB decoded')
        header, separator, encoded = url.partition(',')
        if not separator or header not in ('data:image/png;base64', 'data:image/jpeg;base64'):
            raise ValueError('Expected a PNG/JPEG base64 data URL; URLs and server paths are not accepted')
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError('Invalid image base64') from exc
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError('Each image must be at most 4 MiB decoded')
        if header != f'data:{image_mime(data)};base64':
            raise ValueError('Image MIME type does not match its contents')
    return images


def load_image(path):
    with Path(path).open('rb') as stream:
        data = stream.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError('Each image must be at most 4 MiB')
    return f'data:{image_mime(data)};base64,' + base64.b64encode(data).decode('ascii')
