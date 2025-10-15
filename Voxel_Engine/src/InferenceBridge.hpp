#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <chrono>
#include <atomic>
#include <unordered_map>
#include "../third_party/json.hpp"
#include "../third_party/zmq.hpp"
#include <cstring>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/types.h>
#endif

class InferenceBridge {
public:
    explicit InferenceBridge(const std::string& endpoint)
        : context(1), socket(context, zmq::socket_type::req), endpoint(endpoint) {}

    void connect() {
        socket.connect(endpoint);
        std::cout << "[InferenceBridge] Connected to " << endpoint << std::endl;
    }

    std::vector<float> infer(const std::vector<int32_t>& coords,
                             const std::vector<float>& feats,
                             size_t ncoords,
                             size_t nfeats)
    {
        if (ncoords == 0) return {};

        const size_t coord_dim = 4;
        const size_t coords_bytes = ncoords * coord_dim * sizeof(int32_t);
        const size_t feats_bytes  = ncoords * nfeats   * sizeof(float);

        // --- Unique shm names per call ---
        std::string coords_name = makeShmName("coords");
        std::string feats_name  = makeShmName("feats");
        std::string out_name    = makeShmName("out");

        // --- Create and fill shared memory ---
        void* coords_mem = createSharedMemory(coords_name, coords_bytes);
        void* feats_mem  = createSharedMemory(feats_name,  feats_bytes);

        std::memcpy(coords_mem, coords.data(), coords_bytes);
        std::memcpy(feats_mem,  feats.data(),  feats_bytes);

        // --- Send metadata to Python ---
        nlohmann::json meta = {
            {"coords_shm", coords_name},
            {"feats_shm",  feats_name},
            {"out_shm",    out_name},
            {"coords_shape", {ncoords, coord_dim}},
            {"feats_shape",  {ncoords, nfeats}}
        };
        
        const std::string meta_str = meta.dump();
        socket.send(zmq::buffer(meta_str), zmq::send_flags::none);

        zmq::message_t reply;
        auto r = socket.recv(reply, zmq::recv_flags::none);
        if (!r) throw std::runtime_error("ZMQ recv failed");

        const std::string reply_str(static_cast<const char*>(reply.data()), reply.size());
        std::cout << "[C++] Raw reply from Python: " << reply_str << std::endl;

        nlohmann::json resp = nlohmann::json::parse(reply_str);
        bool ok = resp.value("status", "") == "ok";
        if (!ok) {
            // cleanup before throwing
            destroySharedMemory(coords_name, coords_mem, coords_bytes);
            destroySharedMemory(feats_name,  feats_mem,  feats_bytes);
            const std::string msg = resp.value("message", "unknown");
            throw std::runtime_error("Python inference failed: " + msg);
        }

        size_t out_bytes = resp.value("nbytes", static_cast<size_t>(0));
        if (out_bytes != ncoords * sizeof(float)) {
            // cleanup before throwing
            destroySharedMemory(coords_name, coords_mem, coords_bytes);
            destroySharedMemory(feats_name,  feats_mem,  feats_bytes);
            std::ostringstream oss;
            oss << "Unexpected out_bytes. Expected " << (ncoords * sizeof(float))
                << " got " << out_bytes;
            throw std::runtime_error(oss.str());
        }

        // --- Read output shared memory created by Python ---
        void* out_mem = openSharedMemory(out_name, out_bytes);

        std::vector<float> out(ncoords);
        std::memcpy(out.data(), out_mem, out_bytes);

        // Cleanup
        destroySharedMemory(coords_name, coords_mem, coords_bytes);
        destroySharedMemory(feats_name,  feats_mem,  feats_bytes);
        destroySharedMemory(out_name,    out_mem,   out_bytes);

        return out;
    }

private:
    zmq::context_t context;
    zmq::socket_t socket;
    std::string endpoint;

    static std::string makeShmName(const char* base) {
        static std::atomic<uint64_t> counter{0};
        uint64_t c = ++counter;
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::ostringstream oss;
    #ifdef _WIN32
        oss << "ares_" << base << "_" << GetCurrentProcessId() << "_" << now << "_" << c;
    #else
        oss << "ares_" << base << "_" << getpid() << "_" << now << "_" << c;
    #endif
        return oss.str();
    }

#ifdef _WIN32
    void* createSharedMemory(const std::string& name, size_t size) {
        HANDLE hMapFile = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 
                                             0, static_cast<DWORD>(size), name.c_str());
        if (!hMapFile) throw std::runtime_error("CreateFileMappingA failed (" + name + ")");
        void* ptr = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, size);
        if (!ptr) {
            CloseHandle(hMapFile);
            throw std::runtime_error("MapViewOfFile failed (" + name + ")");
        }
        handle_map[name] = hMapFile;
        return ptr;
    }

    void* openSharedMemory(const std::string& name, size_t size) {
        HANDLE hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
        if (!hMapFile) throw std::runtime_error("OpenFileMappingA failed (" + name + ")");
        void* ptr = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, size);
        if (!ptr) {
            CloseHandle(hMapFile);
            throw std::runtime_error("MapViewOfFile failed (" + name + ")");
        }
        handle_map[name] = hMapFile;
        return ptr;
    }

    void destroySharedMemory(const std::string& name, void* ptr, size_t) {
        if (ptr) UnmapViewOfFile(ptr);
        auto it = handle_map.find(name);
        if (it != handle_map.end()) {
            CloseHandle(it->second);
            handle_map.erase(it);
        }
    }

    std::unordered_map<std::string, HANDLE> handle_map;
#else
    void* createSharedMemory(const std::string& name, size_t size) {
        int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd == -1) throw std::runtime_error("shm_open failed (" + name + ")");
        if (ftruncate(fd, size) == -1) {
            close(fd);
            throw std::runtime_error("ftruncate failed (" + name + ")");
        }
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (ptr == MAP_FAILED) throw std::runtime_error("mmap failed (" + name + ")");
        return ptr;
    }

    void* openSharedMemory(const std::string& name, size_t size) {
        int fd = shm_open(name.c_str(), O_RDWR, 0666);
        if (fd == -1) throw std::runtime_error("shm_open open failed (" + name + ")");
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (ptr == MAP_FAILED) throw std::runtime_error("mmap open failed (" + name + ")");
        return ptr;
    }

    void destroySharedMemory(const std::string& name, void* ptr, size_t size) {
        if (ptr && ptr != MAP_FAILED) munmap(ptr, size);
        shm_unlink(name.c_str());
    }
#endif
};
