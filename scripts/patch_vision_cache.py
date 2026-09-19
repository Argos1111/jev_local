#!/usr/bin/env python3
"""Apply the encoder-only cache to an unmodified llama.cpp b11042 source tree."""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    folder = args.source/'tools/mtmd'
    path = folder/'clip.cpp'
    text = path.read_text()
    if 'jev_vision_embedding_cache' in text:
        raise SystemExit('Cache patch already applied')
    original = '''    return clip_encode(ctx, &params);
}

// persisted state slots'''
    replacement = '''    if (ctx->model.proj_type != PROJECTOR_TYPE_LFM2 || imgs_c_ptr->is_audio || !ctx->jev_cache.enabled()) {
        return clip_encode(ctx, &params);
    }
    // Key the actual preprocessed pixels, shape, and per-tile metadata, NOT
    // the image ID (all tiles of one source image can share the same ID).
    std::string key;
    auto append = [&](const auto & value) {
        key.append(reinterpret_cast<const char *>(&value), sizeof(value));
    };
    append(imgs_c_ptr->entries.size());
    for (const auto & image : imgs_c_ptr->entries) {
        if (image.is_placeholder()) return clip_encode(ctx, &params);
        append(image.nx()); append(image.ny());
        append(image.add_viewsep); append(image.add_newline); append(image.lead_pad);
        const auto & pixels = image.get_ro_buf();
        append(pixels.size());
        key.append(reinterpret_cast<const char *>(pixels.data()), pixels.size() * sizeof(float));
    }
    return ctx->jev_cache.run(std::move(key), out_batch_embd, [&]() { return clip_encode(ctx, &params); });
}

// persisted state slots'''
    if text.count(original) != 1 or text.count('struct clip_ctx {') != 1:
        raise SystemExit('Unexpected source layout; use the pinned b11042 source')
    text = text.replace('#include "clip.h"', '#include "clip.h"\n#include "vision_embedding_cache.h"', 1)
    text = text.replace('struct clip_ctx {', 'struct clip_ctx {\n    jev_vision_embedding_cache jev_cache;', 1)
    text = text.replace(original, replacement, 1)
    shutil.copyfile(Path(__file__).resolve().parents[1]/'native/vision_embedding_cache.h', folder/'vision_embedding_cache.h')
    path.write_text(text)
    build_info = args.source/'cmake/build-info.cmake'
    with build_info.open('a') as stream:
        stream.write('\n# Do not report the enclosing project git revision for the source archive.\n'
                     'set(BUILD_NUMBER 11042)\nset(BUILD_COMMIT "ec9281505-jev-vision-cache")\n')


if __name__ == '__main__':
    main()
