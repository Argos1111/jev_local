#include "vision_embedding_cache.h"
#include <atomic>
#include <cassert>
#include <thread>
#include <chrono>
int main() {
    jev_vision_embedding_cache cache(1024);
    std::atomic<int> calls{0};
    auto request = [&]() {
        std::vector<float> out;
        bool ok = cache.run("image-A/tile-1", out, [&]() {
            ++calls;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            out = {1, 2, 3}; return true;
        });
        assert(ok && out == std::vector<float>({1, 2, 3}));
    };
    std::thread a(request), b(request), c(request), d(request);
    a.join(); b.join(); c.join(); d.join();
    assert(calls == 1);
    std::vector<float> out;
    assert(cache.run("image-A/tile-2", out, [&]() {++calls; out={4}; return true;}));
    assert(calls == 2 && out[0] == 4); // distinct tiles cannot alias
    assert(!cache.run("failed", out, [&]() {++calls; return false;}));
    assert(cache.run("failed", out, [&]() {++calls; out={5}; return true;}));
    assert(calls == 4); // failure was not cached
    jev_vision_embedding_cache tiny(5);
    int small_calls = 0;
    for (auto key : {"a", "b", "a"}) {
        assert(tiny.run(key, out, [&]() {++small_calls; out={1}; return true;}));
    }
    assert(small_calls == 3); // eviction
    int large_calls = 0;
    for (int i=0; i<2; ++i) {
        assert(tiny.run("too large", out, [&]() {++large_calls; out={1,2}; return true;}));
    }
    assert(large_calls == 2); // oversized entries are never retained
    jev_vision_embedding_cache other(1024);
    assert(other.run("image-A/tile-1", out, [&]() {++calls; out={6}; return true;}));
    assert(calls == 5 && out[0] == 6); // different model context never shares
    assert(!jev_vision_embedding_cache(0).enabled());
}
