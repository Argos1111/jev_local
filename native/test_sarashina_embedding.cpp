// Diagnostic only: dump normalized pixels and projected embeddings for comparison.
#include "clip.h"
#include "clip-model.h"
#include "mtmd-image.h"
#include "ggml-backend.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

int main(int argc, char ** argv) {
    if (argc != 8 && !(argc == 9 && std::string(argv[8]) == "pixels-only")) {
        std::cerr << "usage: test_sarashina_embedding MMPROJ RGB WIDTH HEIGHT OUTPUT DEVICE FA [pixels-only]\n";
        return 2;
    }
    try {
        const int width = std::stoi(argv[3]);
        const int height = std::stoi(argv[4]);
        if (width < 1 || height < 1 || width > 4096 || height > 4096) {
            throw std::runtime_error("invalid dimensions");
        }
        std::vector<uint8_t> rgb(size_t(width) * height * 3);
        std::ifstream input(argv[2], std::ios::binary);
        if (!input.read(reinterpret_cast<char *>(rgb.data()), rgb.size())) {
            throw std::runtime_error("cannot read raw RGB");
        }
        ggml_backend_load_all();
        clip_context_params params{};
        const bool pixels_only = argc == 9;
        params.no_alloc = pixels_only;
        const std::string device = argv[6];
        params.use_gpu = device != "CPU";
        params.device = ggml_backend_dev_by_name(device.c_str());
        if (!params.device) {
            throw std::runtime_error("device not found");
        }
        const int fa = std::stoi(argv[7]);
        if (fa < -1 || fa > 1) {
            throw std::runtime_error("invalid FA mode");
        }
        params.flash_attn_type = clip_flash_attn_type(fa);
        params.image_min_tokens = -1;
        params.image_max_tokens = -1;
        auto result = clip_init(argv[1], params);
        std::unique_ptr<clip_ctx, decltype(&clip_free)> owner(result.ctx_v, clip_free);
        auto * ctx = owner.get();
        if (!ctx) {
            throw std::runtime_error("projector load failed");
        }
        clip_image_u8 image;
        image.set_size({width, height}, false);
        image.cpy_buf(rgb);
        mtmd_image_preprocessor_dyn_size preprocessor(ctx);
        auto preprocessed = preprocessor.preprocess(image);
        clip_image_f32_batch batch;
        batch.entries = std::move(preprocessed.entries);
        if (batch.entries.size() != 1) {
            throw std::runtime_error("expected one preprocessed image");
        }
        std::vector<float> embeddings;
        if (!pixels_only) {
            embeddings.resize(size_t(clip_n_output_tokens(ctx, &batch.entries[0])) * clip_n_mmproj_embd(ctx));
            if (!clip_image_batch_encode(ctx, 8, &batch, embeddings)) {
                throw std::runtime_error("encoding failed");
            }
        }
        auto save = [&](const std::string & suffix, const std::vector<float> & values) {
            std::ofstream output(std::string(argv[5]) + suffix, std::ios::binary);
            if (!output.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float))) {
                throw std::runtime_error("cannot write diagnostic output");
            }
        };
        save(".pixels.f32", batch.entries[0].get_ro_buf());
        if (!pixels_only) {
            save(".embd.f32", embeddings);
        }
        std::cout << "{\"width\":" << batch.entries[0].nx()
                  << ",\"height\":" << batch.entries[0].ny()
                  << ",\"tokens\":" << clip_n_output_tokens(ctx, &batch.entries[0])
                  << ",\"embedding_dim\":" << clip_n_mmproj_embd(ctx) << "}\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
