#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <vector>
#include <utility>
#include <string>
#include <cstring>
#include <stdexcept>

class IntersectionEnv;
struct EnvState;

namespace py = pybind11;

// ============================================================================
// 优化数据结构
// ============================================================================

/**
 * RingBuffer: 环形缓冲区，避免shift_append_obs_seq中的内存拷贝
 * 用于管理观测序列，支持O(1)时间复杂度的追加操作
 */
class RingBuffer {
public:
    RingBuffer() : seq_len_(0), obs_dim_(0), head_(0) {}
    
    RingBuffer(int seq_len, int obs_dim) 
        : seq_len_(seq_len), obs_dim_(obs_dim), head_(0) {
        data_.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(obs_dim), 0.0f);
    }
    
    void init(int seq_len, int obs_dim) {
        seq_len_ = seq_len;
        obs_dim_ = obs_dim;
        head_ = 0;
        data_.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(obs_dim), 0.0f);
    }
    
    void reset() {
        head_ = 0;
        std::fill(data_.begin(), data_.end(), 0.0f);
    }
    
    // 用单个观测填充整个序列（用于初始化）
    void fill_with(const float* obs) {
        for (int t = 0; t < seq_len_; ++t) {
            std::memcpy(data_.data() + static_cast<size_t>(t) * static_cast<size_t>(obs_dim_),
                       obs, sizeof(float) * static_cast<size_t>(obs_dim_));
        }
        head_ = 0;
    }
    
    void fill_with(const std::vector<float>& obs) {
        if (static_cast<int>(obs.size()) != obs_dim_) {
            throw std::runtime_error("RingBuffer::fill_with: obs size mismatch");
        }
        fill_with(obs.data());
    }
    
    // O(1)追加新观测，覆盖最旧的
    void append(const float* obs_new) {
        std::memcpy(data_.data() + static_cast<size_t>(head_) * static_cast<size_t>(obs_dim_),
                   obs_new, sizeof(float) * static_cast<size_t>(obs_dim_));
        head_ = (head_ + 1) % seq_len_;
    }
    
    void append(const std::vector<float>& obs_new) {
        if (static_cast<int>(obs_new.size()) != obs_dim_) {
            throw std::runtime_error("RingBuffer::append: obs size mismatch");
        }
        append(obs_new.data());
    }
    
    // 输出为扁平化序列（按时间顺序：最旧到最新）
    void to_flat(std::vector<float>& out) const {
        out.resize(static_cast<size_t>(seq_len_) * static_cast<size_t>(obs_dim_));
        for (int i = 0; i < seq_len_; ++i) {
            int idx = (head_ + i) % seq_len_;
            std::memcpy(out.data() + static_cast<size_t>(i) * static_cast<size_t>(obs_dim_),
                       data_.data() + static_cast<size_t>(idx) * static_cast<size_t>(obs_dim_),
                       sizeof(float) * static_cast<size_t>(obs_dim_));
        }
    }
    
    std::vector<float> to_flat() const {
        std::vector<float> out;
        to_flat(out);
        return out;
    }
    
    // 从扁平化序列初始化
    void from_flat(const std::vector<float>& flat) {
        if (static_cast<int>(flat.size()) != seq_len_ * obs_dim_) {
            throw std::runtime_error("RingBuffer::from_flat: size mismatch");
        }
        data_ = flat;
        head_ = 0;
    }
    
    int seq_len() const { return seq_len_; }
    int obs_dim() const { return obs_dim_; }
    bool empty() const { return data_.empty(); }
    
private:
    int seq_len_;
    int obs_dim_;
    int head_;  // 指向下一个要写入的位置（也是最旧数据的位置）
    std::vector<float> data_;
};

/**
 * 预分配的工作缓冲区，避免rollout中频繁的内存分配
 */
struct RolloutWorkspace {
    // 其他智能体的观测序列缓冲区
    std::vector<RingBuffer> other_obs_bufs;
    // 其他智能体的LSTM隐藏状态
    std::vector<std::vector<float>> other_h;
    std::vector<std::vector<float>> other_c;
    // 临时张量缓冲区
    std::vector<float> temp_obs_flat;
    std::vector<float> throttles;
    std::vector<float> steerings;
    // 是否已初始化
    bool initialized = false;
    
    void init(int n_agents, int seq_len, int obs_dim, int lstm_hidden_dim) {
        other_obs_bufs.resize(static_cast<size_t>(n_agents));
        other_h.resize(static_cast<size_t>(n_agents));
        other_c.resize(static_cast<size_t>(n_agents));
        
        for (int j = 0; j < n_agents; ++j) {
            other_obs_bufs[static_cast<size_t>(j)].init(seq_len, obs_dim);
            other_h[static_cast<size_t>(j)].resize(static_cast<size_t>(lstm_hidden_dim), 0.0f);
            other_c[static_cast<size_t>(j)].resize(static_cast<size_t>(lstm_hidden_dim), 0.0f);
        }
        
        temp_obs_flat.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(obs_dim));
        throttles.resize(static_cast<size_t>(n_agents));
        steerings.resize(static_cast<size_t>(n_agents));
        initialized = true;
    }
    
    void reset(int n_agents) {
        for (int j = 0; j < n_agents; ++j) {
            other_obs_bufs[static_cast<size_t>(j)].reset();
            std::fill(other_h[static_cast<size_t>(j)].begin(), 
                     other_h[static_cast<size_t>(j)].end(), 0.0f);
            std::fill(other_c[static_cast<size_t>(j)].begin(), 
                     other_c[static_cast<size_t>(j)].end(), 0.0f);
        }
    }
};

// 线程局部的工作空间，避免多线程竞争
#ifdef _MSC_VER
    #define MCTS_THREAD_LOCAL __declspec(thread)
#else
    #define MCTS_THREAD_LOCAL thread_local
#endif

// ============================================================================
// MCTS搜索函数声明
// ============================================================================

std::pair<std::vector<float>, py::dict> mcts_search_lstm_torchscript(
    IntersectionEnv& env,
    const EnvState& root_state,
    const std::vector<float>& root_obs_seq, // flattened (T*obs_dim)
    int seq_len,
    int obs_dim,
    const std::string& model_path,
    const std::vector<float>& root_h,
    const std::vector<float>& root_c,
    int lstm_hidden_dim,
    int agent_index,
    int num_simulations,
    int num_action_samples,
    int rollout_depth,
    float c_puct,
    float temperature,
    float gamma,
    float dirichlet_alpha,
    float dirichlet_eps,
    unsigned int seed
);

std::pair<std::vector<float>, py::dict> mcts_search(
    IntersectionEnv& env,
    const EnvState& root_state,
    const std::vector<float>& root_obs,
    const py::function& infer_fn,
    int agent_index,
    int num_simulations,
    int num_action_samples,
    int rollout_depth,
    float c_puct,
    float temperature,
    float gamma,
    float dirichlet_alpha,
    float dirichlet_eps,
    unsigned int seed
);

// (LSTM) MCTS API (2-step).
// infer_policy_value signature (Python):
//   (obs_batch: List[List[float]], h_batch: np.ndarray(B,H) or (B,1,H), c_batch: np.ndarray(B,H) or (B,1,H))
//     -> (mean: np.ndarray(B,2), std: np.ndarray(B,2), value: np.ndarray(B) or (B,1))
// infer_next_hidden signature (Python):
//   (obs_batch: List[List[float]], h_batch: np.ndarray(B,H) or (B,1,H), c_batch: np.ndarray(B,H) or (B,1,H), action_batch: np.ndarray(B,2))
//     -> (h_next: np.ndarray(B,H) or (B,1,H), c_next: np.ndarray(B,H) or (B,1,H))
std::pair<std::vector<float>, py::dict> mcts_search_lstm(
    IntersectionEnv& env,
    const EnvState& root_state,
    const std::vector<float>& root_obs,
    const py::function& infer_policy_value,
    const py::function& infer_next_hidden,
    const std::vector<float>& root_h,
    const std::vector<float>& root_c,
    int lstm_hidden_dim,
    int agent_index,
    int num_simulations,
    int num_action_samples,
    int rollout_depth,
    float c_puct,
    float temperature,
    float gamma,
    float dirichlet_alpha,
    float dirichlet_eps,
    unsigned int seed
);