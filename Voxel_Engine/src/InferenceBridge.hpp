#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
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
        // Shared memory names
        std::string coords_name = "coords_shm";
        std::string feats_name  = "feats_shm";
        std::string out_name    = "out_shm";

        size_t coords_bytes = ncoords * 4 * sizeof(int32_t);
        size_t feats_bytes  = ncoords * nfeats * sizeof(float);

        // --- Create shared memory blocks ---
        void* coords_mem = createSharedMemory(coords_name, coords_bytes);
        void* feats_mem  = createSharedMemory(feats_name, feats_bytes);

        std::memcpy(coords_mem, coords.data(), coords_bytes);
        std::memcpy(feats_mem, feats.data(), feats_bytes);

        // --- Send metadata to Python ---
        nlohmann::json meta = {
            {"coords_shm", coords_name},
            {"feats_shm", feats_name},
            {"out_shm", out_name},
            {"coords_shape", {ncoords, 4}},
            {"feats_shape", {ncoords, nfeats}}
        };

        socket.send(zmq::buffer(meta.dump()), zmq::send_flags::none);

        zmq::message_t reply;
        (void)socket.recv(reply, zmq::recv_flags::none);

        std::string replyStr = reply.to_string();
        std::cerr << "[C++] Raw reply from Python: " << replyStr << std::endl;

        nlohmann::json resp;
        try {
            resp = nlohmann::json::parse(replyStr);
        } catch (const std::exception& e) {
            cleanupSharedMemory(coords_name, coords_mem, coords_bytes,
                                feats_name, feats_mem, feats_bytes,
                                out_name, nullptr, 0);
            throw std::runtime_error(std::string("Invalid JSON from Python: ") + e.what());
        }

        // --- Validate and handle response ---
        if (resp.contains("status") && resp["status"].is_string()) {
            std::string status = resp["status"];

            if (status == "ok") {
                if (!resp.contains("nbytes") || !resp["nbytes"].is_number()) {
                    cleanupSharedMemory(coords_name, coords_mem, coords_bytes,
                                        feats_name, feats_mem, feats_bytes,
                                        out_name, nullptr, 0);
                    throw std::runtime_error("Missing or invalid 'nbytes' in Python reply");
                }

                size_t out_bytes = resp["nbytes"].get<size_t>();
                void* out_mem = openSharedMemory(out_name, out_bytes);

                std::vector<float> out(ncoords);
                std::memcpy(out.data(), out_mem, out_bytes);

                cleanupSharedMemory(coords_name, coords_mem, coords_bytes,
                                    feats_name, feats_mem, feats_bytes,
                                    out_name, out_mem, out_bytes);

                return out;
            }

            std::string msg = resp.value("message", "Unknown error from Python");
            cleanupSharedMemory(coords_name, coords_mem, coords_bytes,
                                feats_name, feats_mem, feats_bytes,
                                out_name, nullptr, 0);
            throw std::runtime_error("Python inference failed: " + msg);
        }

        cleanupSharedMemory(coords_name, coords_mem, coords_bytes,
                            feats_name, feats_mem, feats_bytes,
                            out_name, nullptr, 0);
        throw std::runtime_error("Invalid JSON: missing or non-string 'status'");
    }

private:
    zmq::context_t context;
    zmq::socket_t socket;
    std::string endpoint;

#ifdef _WIN32
    void* createSharedMemory(const std::string& name, size_t size) {
        HANDLE hMapFile = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, (DWORD)size, name.c_str());
        if (!hMapFile) throw std::runtime_error("CreateFileMappingA failed");
        void* ptr = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, size);
        if (!ptr) throw std::runtime_error("MapViewOfFile failed");
        handle_map[name] = hMapFile;
        return ptr;
    }

    void* openSharedMemory(const std::string& name, size_t size) {
        HANDLE hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
        if (!hMapFile) throw std::runtime_error("OpenFileMappingA failed");
        void* ptr = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, size);
        if (!ptr) throw std::runtime_error("MapViewOfFile failed");
        handle_map[name] = hMapFile;
        return ptr;
    }

    void destroySharedMemory(const std::string& name, void* ptr, size_t) {
        UnmapViewOfFile(ptr);
        if (handle_map.count(name)) {
            CloseHandle(handle_map[name]);
            handle_map.erase(name);
        }
    }

    std::unordered_map<std::string, HANDLE> handle_map;
#else
    void* createSharedMemory(const std::string& name, size_t size) {
        int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd == -1) throw std::runtime_error("shm_open failed");
        if (ftruncate(fd, size) == -1) throw std::runtime_error("ftruncate failed");
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) throw std::runtime_error("mmap failed");
        close(fd);
        return ptr;
    }

    void* openSharedMemory(const std::string& name, size_t size) {
        int fd = shm_open(name.c_str(), O_RDWR, 0666);
        if (fd == -1) throw std::runtime_error("shm_open open failed");
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) throw std::runtime_error("mmap open failed");
        close(fd);
        return ptr;
    }

    void destroySharedMemory(const std::string& name, void* ptr, size_t size) {
        munmap(ptr, size);
        shm_unlink(name.c_str());
    }
#endif

    void cleanupSharedMemory(const std::string& n1, void* p1, size_t s1,
                             const std::string& n2, void* p2, size_t s2,
                             const std::string& n3, void* p3, size_t s3) {
        try {
            destroySharedMemory(n1, p1, s1);
        } catch (...) {}
        try {
            destroySharedMemory(n2, p2, s2);
        } catch (...) {}
        try {
            if (p3) destroySharedMemory(n3, p3, s3);
        } catch (...) {}
    }
};
