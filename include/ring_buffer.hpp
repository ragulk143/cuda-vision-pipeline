#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : buffer_(capacity), capacity_(capacity) {}

    // Non-blocking push: drops oldest item on overflow so latency stays bounded
    // Returns true if an older item had to be dropped
    bool push(T&& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (closed_) return false;

        bool dropped = false;
        if (size_ == capacity_) {
            head_ = (head_ + 1) % capacity_;
            --size_;
            ++dropped_count_;
            dropped = true;
        }

        buffer_[tail_] = std::move(item);
        tail_ = (tail_ + 1) % capacity_;
        ++size_;

        lock.unlock();
        cv_not_empty_.notify_one();
        return dropped;
    }

    // Blocking pop until an item is available or queue is closed & empty
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_not_empty_.wait(lock, [this]() { return size_ > 0 || closed_; });

        if (size_ == 0 && closed_) {
            return std::nullopt;
        }

        T item = std::move(buffer_[head_]);
        head_ = (head_ + 1) % capacity_;
        --size_;
        return item;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            closed_ = true;
        }
        cv_not_empty_.notify_all();
    }

    size_t dropped_count() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return dropped_count_;
    }

private:
    std::vector<T> buffer_;
    size_t capacity_;
    size_t head_ = 0;
    size_t tail_ = 0;
    size_t size_ = 0;
    size_t dropped_count_ = 0;
    bool closed_ = false;
    mutable std::mutex mtx_;
    std::condition_variable cv_not_empty_;
};
