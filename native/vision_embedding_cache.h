#pragma once
// Cache only the vision encoder/projector output; no decoder/KV/state reuse.
#include <list>
#include <mutex>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

class jev_vision_embedding_cache {
    struct entry { std::string key; std::vector<float> embedding; };
    std::list<entry> lru;
    size_t used = 0;
    size_t limit;
    std::mutex mutex;
    unsigned long hits = 0, misses = 0;
public:
    static size_t configured_limit() {
        const char * value = std::getenv("JEV_IMAGE_CACHE_MIB");
        if (!value || !*value) return 0; // opt-in; stock behavior by default
        char * end = nullptr;
        long mib = std::strtol(value, &end, 10);
        if (*end || mib < 0 || mib > 4096) throw std::runtime_error("JEV_IMAGE_CACHE_MIB must be 0..4096");
        return static_cast<size_t>(mib) * 1024 * 1024;
    }
    explicit jev_vision_embedding_cache(size_t bytes = configured_limit()) : limit(bytes) {}
    bool enabled() const { return limit != 0; }
    template<class Compute>
    bool run(std::string key, std::vector<float> & out, Compute compute) {
        // A context has one encoder. Hold the lock through computation so parallel
        // identical misses encode only once and cannot race encoder graph buffers.
        std::lock_guard<std::mutex> guard(mutex);
        for (auto it = lru.begin(); it != lru.end(); ++it) {
            if (it->key == key) { // exact float bytes + shape: no hash/tile-ID collisions
                out = it->embedding;
                lru.splice(lru.begin(), lru, it);
                ++hits;
                std::fprintf(stderr, "JEV_IMAGE_CACHE hit hits=%lu misses=%lu bytes=%zu\n", hits, misses, used);
                return true;
            }
        }
        ++misses;
        if (!compute()) return false; // never cache partial/failed output
        const size_t bytes = key.size() + out.size() * sizeof(float);
        if (bytes <= limit) {
            while (used > limit - bytes && !lru.empty()) {
                used -= lru.back().key.size() + lru.back().embedding.size() * sizeof(float);
                lru.pop_back();
            }
            lru.push_front({std::move(key), out});
            used += bytes;
        }
        std::fprintf(stderr, "JEV_IMAGE_CACHE miss hits=%lu misses=%lu bytes=%zu\n", hits, misses, used);
        return true;
    }
};
