#pragma once
#include <vector>
#include <mutex>
#include <condition_variable>
#include <optional>

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : buffer_(capacity), capacity_(capacity) {}

    void push(T&& item, size_t& dropped_frames) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (count_ == capacity_) {
            head_ = (head_ + 1) % capacity_;
            --count_;
            ++dropped_frames;
        }
        buffer_[tail_] = std::move(item);
        tail_ = (tail_ + 1) % capacity_;
        ++count_;
        cv_not_empty_.notify_one();
    }

    std::optional<T> pop(bool& running) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_not_empty_.wait(lock, [&]() { return count_ > 0 || !running; });
        if (count_ == 0 && !running) {
            return std::nullopt;
        }
        T item = std::move(buffer_[head_]);
        head_ = (head_ + 1) % capacity_;
        --count_;
        return item;
    }

    void notify_all() { cv_not_empty_.notify_all(); }

private:
    std::vector<T> buffer_;
    size_t capacity_;
    size_t head_ = 0;
    size_t tail_ = 0;
    size_t count_ = 0;
    std::mutex mtx_;
    std::condition_variable cv_not_empty_;
};
