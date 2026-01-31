// ============================================================================
// mcts_search.cpp - 优化版本
// 主要优化：
// 1. 确定性模拟策略 - rollout中所有智能体在t>0时使用均值动作
// 2. RingBuffer替代shift_append_obs_seq减少内存拷贝
// 3. 预分配工作空间减少动态内存分配
// 4. 模型缓存优化
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <torch/script.h>

#include <cstring>

#include <cmath>
#include <random>
#include <vector>
#include <utility>
#include <algorithm>
#include <limits>
#include <array>

#include "IntersectionEnv.h"
#include "EnvState.h"
#include "mcts_search.h"

#include <string>
#include <mutex>
#include <unordered_map>
#include <filesystem>
#include <chrono>

namespace py = pybind11;

// ============================================================================
// 性能分析结构
// ============================================================================

struct TSProfile {
    double ms_infer_pv{0.0};
    double ms_infer_nh{0.0};
    double ms_env_step{0.0};
    double ms_total{0.0};
    uint64_t calls_infer_pv{0};
    uint64_t calls_infer_nh{0};
    uint64_t calls_env_step{0};
};

static inline double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// ============================================================================
// 节点和边结构（与原版相同）
// ============================================================================

struct ChildEdge {
    std::array<float, 2> action;
    float prior{0.0f};
    int N{0};
    float W{0.0f};
    int child{-1};
};

struct Node {
    EnvState state;
    float V{0.0f};
    int total_N{0};
    std::vector<ChildEdge> edges;
};

struct LSTMEdge {
    std::array<float, 2> action;
    float prior{0.0f};
    int N{0};
    float W{0.0f};
    int child{-1};
    std::vector<float> h_next;
    std::vector<float> c_next;
};

struct LSTMNode {
    EnvState state;
    float V{0.0f};
    int total_N{0};
    std::vector<LSTMEdge> edges;
    std::vector<float> h;
    std::vector<float> c;
    std::vector<float> obs_seq_flat;
};

// ============================================================================
// 工具函数
// ============================================================================

static inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

static inline float randn(std::mt19937& rng) {
    static thread_local std::normal_distribution<float> nd(0.0f, 1.0f);
    return nd(rng);
}

static inline float gaussian_log_prob(float x, float mean, float std) {
    if (std <= 1e-6f) return -std::numeric_limits<float>::infinity();
    const float var = std * std;
    const float log_std = std::log(std);
    constexpr float log_2pi = 1.83787706641f;
    return -0.5f * ((x - mean) * (x - mean) / var + log_2pi) - log_std;
}

static void softmax_inplace(std::vector<float>& logits) {
    if (logits.empty()) return;
    float max_logit = -std::numeric_limits<float>::infinity();
    for (float v : logits) if (v > max_logit) max_logit = v;

    float sum = 0.0f;
    for (float& v : logits) {
        v = std::exp(v - max_logit);
        sum += v;
    }
    if (sum > 0.0f) {
        for (float& v : logits) v /= sum;
    }
}

static void parse_infer_out(const py::tuple& out,
                            std::vector<std::array<float,2>>& means,
                            std::vector<std::array<float,2>>& stds,
                            std::vector<float>& values) {
    py::array mean_arr = py::array::ensure(out[0]);
    py::array std_arr  = py::array::ensure(out[1]);
    py::array val_arr  = py::array::ensure(out[2]);

    auto m = mean_arr.unchecked<float, 2>();
    auto s = std_arr.unchecked<float, 2>();

    size_t B = (size_t)m.shape(0);
    means.resize(B);
    stds.resize(B);
    values.resize(B);

    if (val_arr.ndim() == 2) {
        auto v = val_arr.unchecked<float, 2>();
        for (size_t i = 0; i < B; ++i) {
            means[i] = { m(i,0), m(i,1) };
            stds[i]  = { s(i,0), s(i,1) };
            values[i] = v(i,0);
        }
    } else {
        auto v = val_arr.unchecked<float, 1>();
        for (size_t i = 0; i < B; ++i) {
            means[i] = { m(i,0), m(i,1) };
            stds[i]  = { s(i,0), s(i,1) };
            values[i] = v(i);
        }
    }
}

static void parse_hc_batch(const py::handle& arr_h,
                           int lstm_hidden_dim,
                           std::vector<std::vector<float>>& out_batch) {
    py::array a = py::array::ensure(arr_h);
    if (!a) throw std::runtime_error("h/c must be array-like");

    if (a.ndim() == 2) {
        if ((int)a.shape(1) != lstm_hidden_dim) {
            throw std::runtime_error("h/c shape mismatch: expected second dim H");
        }
        auto u = a.unchecked<float, 2>();
        size_t B = (size_t)u.shape(0);
        out_batch.assign(B, std::vector<float>((size_t)lstm_hidden_dim, 0.0f));
        for (size_t b = 0; b < B; ++b) {
            for (int i = 0; i < lstm_hidden_dim; ++i) out_batch[b][(size_t)i] = u(b, i);
        }
        return;
    }

    if (a.ndim() == 3) {
        if ((int)a.shape(1) != 1 || (int)a.shape(2) != lstm_hidden_dim) {
            throw std::runtime_error("h/c shape mismatch: expected (B,1,H)");
        }
        auto u = a.unchecked<float, 3>();
        size_t B = (size_t)u.shape(0);
        out_batch.assign(B, std::vector<float>((size_t)lstm_hidden_dim, 0.0f));
        for (size_t b = 0; b < B; ++b) {
            for (int i = 0; i < lstm_hidden_dim; ++i) out_batch[b][(size_t)i] = u(b, 0, i);
        }
        return;
    }

    throw std::runtime_error("h/c must be 2D (B,H) or 3D (B,1,H)");
}

static float puct_score_basic(int total_N, float prior, int N, float W, float c_puct) {
    const float Q = (N > 0) ? (W / float(N)) : 0.0f;
    const float U = c_puct * prior * std::sqrt(float(std::max(1, total_N))) / (1.0f + float(N));
    return Q + U;
}

// ============================================================================
// TorchScript推理辅助函数
// ============================================================================

static inline torch::Tensor make_obs_seq_tensor(
    const std::vector<float>& obs_seq_flat,
    int seq_len,
    int obs_dim,
    torch::Device device
) {
    if ((int)obs_seq_flat.size() != seq_len * obs_dim) {
        throw std::runtime_error("root_obs_seq size mismatch: expected seq_len*obs_dim");
    }
    auto t = torch::from_blob(
        const_cast<float*>(obs_seq_flat.data()),
        {1, seq_len, obs_dim},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();
    return t.to(device);
}

// 优化：使用RingBuffer的版本
static inline torch::Tensor make_obs_seq_tensor_from_ring(
    const RingBuffer& ring_buf,
    torch::Device device,
    std::vector<float>& temp_flat  // 预分配的临时缓冲区
) {
    ring_buf.to_flat(temp_flat);
    auto t = torch::from_blob(
        temp_flat.data(),
        {1, ring_buf.seq_len(), ring_buf.obs_dim()},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();
    return t.to(device);
}

static inline torch::Tensor make_hc_tensor(
    const std::vector<float>& hc,
    int lstm_hidden_dim,
    torch::Device device
) {
    if ((int)hc.size() != lstm_hidden_dim) {
        std::vector<float> z((size_t)lstm_hidden_dim, 0.0f);
        auto t = torch::from_blob(z.data(), {1, 1, lstm_hidden_dim}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
        return t.to(device);
    }
    auto t = torch::from_blob(
        const_cast<float*>(hc.data()),
        {1, 1, lstm_hidden_dim},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();
    return t.to(device);
}

static inline void tensor_to_vec1d(const torch::Tensor& t_in, std::vector<float>& out) {
    torch::Tensor t = t_in.detach().to(torch::kCPU).contiguous().view({-1});
    out.resize((size_t)t.numel());
    std::memcpy(out.data(), t.data_ptr<float>(), sizeof(float) * (size_t)t.numel());
}

static inline void parse_policy_value_tensors(
    const torch::Tensor& mean_t,
    const torch::Tensor& std_t,
    const torch::Tensor& val_t,
    std::array<float,2>& mean_out,
    std::array<float,2>& std_out,
    float& value_out
) {
    auto m = mean_t.detach().to(torch::kCPU).contiguous().view({-1});
    auto s = std_t.detach().to(torch::kCPU).contiguous().view({-1});
    auto v = val_t.detach().to(torch::kCPU).contiguous().view({-1});
    if (m.numel() < 2 || s.numel() < 2 || v.numel() < 1) {
        throw std::runtime_error("TorchScript infer_policy_value returned bad shapes");
    }
    const float* mp = m.data_ptr<float>();
    const float* sp = s.data_ptr<float>();
    mean_out = {mp[0], mp[1]};
    std_out  = {sp[0], sp[1]};
    value_out = v.data_ptr<float>()[0];
}

// 保留原版shift_append用于兼容性，但新代码优先使用RingBuffer
static inline std::vector<float> shift_append_obs_seq(
    const std::vector<float>& obs_seq_flat,
    int seq_len,
    int obs_dim,
    const std::vector<float>& obs_new
) {
    if ((int)obs_seq_flat.size() != seq_len * obs_dim) {
        throw std::runtime_error("obs_seq_flat size mismatch");
    }
    if ((int)obs_new.size() != obs_dim) {
        throw std::runtime_error("obs_new size mismatch");
    }
    std::vector<float> out((size_t)seq_len * (size_t)obs_dim);
    const size_t stride = (size_t)obs_dim;
    for (int t = 0; t < seq_len - 1; ++t) {
        std::memcpy(
            out.data() + (size_t)t * stride,
            obs_seq_flat.data() + (size_t)(t + 1) * stride,
            sizeof(float) * stride
        );
    }
    std::memcpy(
        out.data() + (size_t)(seq_len - 1) * stride,
        obs_new.data(),
        sizeof(float) * stride
    );
    return out;
}

// ============================================================================
// 模型缓存（线程安全）
// ============================================================================

static std::mutex g_model_cache_mutex;
static std::unordered_map<std::string, torch::jit::Module> g_model_cache;

static torch::jit::Module& get_cached_model(const std::string& path, torch::Device device) {
    std::lock_guard<std::mutex> lock(g_model_cache_mutex);
    
    auto it = g_model_cache.find(path);
    if (it != g_model_cache.end()) {
        return it->second;
    }
    
    auto model = torch::jit::load(path);
    model.to(device);
    model.eval();
    g_model_cache[path] = std::move(model);
    return g_model_cache[path];
}

// ============================================================================
// 线程局部工作空间
// ============================================================================

static MCTS_THREAD_LOCAL RolloutWorkspace t_workspace;

// ============================================================================
// 确定性Rollout函数 - Python回调版本
// 关键优化：t>0时所有智能体（包括controlled agent）都使用均值动作
// ============================================================================

static std::vector<float> step_env_with_joint_action_deterministic(
    IntersectionEnv& env,
    const EnvState& base_state,
    int agent_index,
    const std::array<float,2>& agent_action,  // 仅在t=0时使用
    const py::function& infer_policy_value,
    const py::function* infer_next_hidden,
    const std::vector<float>* agent_h,
    const std::vector<float>* agent_c,
    int lstm_hidden_dim,
    int rollout_depth,
    float gamma,
    bool& done_out,
    EnvState& next_state_out,
    float& total_return_out,
    std::vector<float>* h_out,
    std::vector<float>* c_out,
    TSProfile* prof
) {
    EnvState saved = env.get_state();
    env.set_state(base_state);

    const int n_agents = (int)base_state.cars.size();

    float total_reward = 0.0f;
    float discount = 1.0f;
    bool terminated = false;

    // 受控智能体的LSTM状态
    std::vector<float> h_cur;
    std::vector<float> c_cur;
    if (infer_next_hidden && agent_h && agent_c && lstm_hidden_dim > 0) {
        h_cur = *agent_h;
        c_cur = *agent_c;
        if ((int)h_cur.size() != lstm_hidden_dim) h_cur.assign((size_t)lstm_hidden_dim, 0.0f);
        if ((int)c_cur.size() != lstm_hidden_dim) c_cur.assign((size_t)lstm_hidden_dim, 0.0f);
    }

    // 预分配动作缓冲区
    std::vector<float> throttles((size_t)n_agents, 0.0f);
    std::vector<float> steerings((size_t)n_agents, 0.0f);

    for (int t = 0; t < rollout_depth; ++t) {
        // 获取所有智能体的观测
        auto all_obs = env.get_observations();

        // 批量收集需要推理的观测
        py::list obs_batch;
        std::vector<int> batch_to_agent;
        batch_to_agent.reserve((size_t)n_agents);

        for (int j = 0; j < n_agents; ++j) {
            // 确定性策略：t>0时，所有智能体都需要推理以获取均值动作
            // t=0时，controlled agent使用传入的agent_action
            if (t == 0 && j == agent_index) continue;
            if (j >= (int)all_obs.size()) continue;

            py::list row;
            for (float v : all_obs[(size_t)j]) row.append(v);
            obs_batch.append(row);
            batch_to_agent.push_back(j);
        }

        // 批量推理
        if (!batch_to_agent.empty()) {
            const int B = (int)batch_to_agent.size();
            py::array_t<float> h_arr({B, 1});
            py::array_t<float> c_arr({B, 1});
            {
                auto hm = h_arr.mutable_unchecked<2>();
                auto cm = c_arr.mutable_unchecked<2>();
                for (int b = 0; b < B; ++b) {
                    hm(b, 0) = 0.0f;
                    cm(b, 0) = 0.0f;
                }
            }

            double t0_pv = 0.0;
            if (prof) t0_pv = now_ms();
            py::tuple pv = infer_policy_value(obs_batch, h_arr, c_arr).cast<py::tuple>();
            if (prof) { prof->ms_infer_pv += (now_ms() - t0_pv); prof->calls_infer_pv += 1; }
            
            if (pv.size() < 2) throw std::runtime_error("infer_policy_value must return (mean,std,value)");

            std::vector<std::array<float,2>> means, stds;
            std::vector<float> values;
            parse_infer_out(pv, means, stds, values);

            for (int b = 0; b < B; ++b) {
                const int j = batch_to_agent[(size_t)b];
                // 确定性：使用均值动作
                throttles[(size_t)j] = clampf(means[(size_t)b][0], -1.0f, 1.0f);
                steerings[(size_t)j] = clampf(means[(size_t)b][1], -1.0f, 1.0f);
            }
        }

        // t=0时，controlled agent使用传入的候选动作
        if (t == 0 && agent_index >= 0 && agent_index < n_agents) {
            throttles[(size_t)agent_index] = agent_action[0];
            steerings[(size_t)agent_index] = agent_action[1];
        }

        // 执行环境步进
        double t0_env = 0.0;
        if (prof) t0_env = now_ms();
        auto res = env.step(throttles, steerings);
        if (prof) { prof->ms_env_step += (now_ms() - t0_env); prof->calls_env_step += 1; }

        // 累积受控智能体的奖励
        if ((int)res.rewards.size() > agent_index && agent_index >= 0) {
            total_reward += discount * res.rewards[(size_t)agent_index];
        }

        // 更新受控智能体的LSTM隐藏状态
        if (infer_next_hidden && h_cur.size() == (size_t)lstm_hidden_dim && c_cur.size() == (size_t)lstm_hidden_dim) {
            if (agent_index >= 0 && agent_index < (int)res.obs.size()) {
                py::list obs_b;
                {
                    py::list row;
                    for (float v : res.obs[(size_t)agent_index]) row.append(v);
                    obs_b.append(row);
                }

                py::array_t<float> h_rep({1, lstm_hidden_dim});
                py::array_t<float> c_rep({1, lstm_hidden_dim});
                {
                    auto hm = h_rep.mutable_unchecked<2>();
                    auto cm = c_rep.mutable_unchecked<2>();
                    for (int i = 0; i < lstm_hidden_dim; ++i) {
                        hm(0, i) = h_cur[(size_t)i];
                        cm(0, i) = c_cur[(size_t)i];
                    }
                }

                // 使用实际执行的动作
                py::array_t<float> act_arr({1, 2});
                {
                    auto am = act_arr.mutable_unchecked<2>();
                    am(0, 0) = throttles[(size_t)agent_index];
                    am(0, 1) = steerings[(size_t)agent_index];
                }

                double t0_nh = 0.0;
                if (prof) t0_nh = now_ms();
                py::tuple hc = (*infer_next_hidden)(obs_b, h_rep, c_rep, act_arr).cast<py::tuple>();
                if (prof) { prof->ms_infer_nh += (now_ms() - t0_nh); prof->calls_infer_nh += 1; }
                
                if (hc.size() >= 2) {
                    std::vector<std::vector<float>> h_next_b, c_next_b;
                    parse_hc_batch(hc[0], lstm_hidden_dim, h_next_b);
                    parse_hc_batch(hc[1], lstm_hidden_dim, c_next_b);
                    if (!h_next_b.empty() && !c_next_b.empty()) {
                        h_cur = std::move(h_next_b[0]);
                        c_cur = std::move(c_next_b[0]);
                    }
                }
            }
        }

        discount *= gamma;
        if (res.terminated || res.truncated) { terminated = true; break; }
    }

    next_state_out = env.get_state();

    auto obs = env.get_observations();
    std::vector<float> ego_obs;
    if (agent_index >= 0 && agent_index < (int)obs.size()) ego_obs = obs[(size_t)agent_index];

    env.set_state(saved);

    done_out = terminated;
    total_return_out = total_reward;

    if (h_out && c_out && !h_cur.empty() && !c_cur.empty()) {
        *h_out = std::move(h_cur);
        *c_out = std::move(c_cur);
    }

    return ego_obs;
}

// ============================================================================
// 确定性Rollout函数 - TorchScript版本（优化内存使用）
// ============================================================================

static std::vector<float> step_env_with_joint_action_torchscript_deterministic(
    IntersectionEnv& env,
    const EnvState& base_state,
    int agent_index,
    const std::array<float,2>& agent_action,  // 仅在t=0时使用
    torch::jit::Module& module,
    const torch::Device& device,
    std::vector<float> obs_seq_flat,
    std::vector<float> h_cur,
    std::vector<float> c_cur,
    int seq_len,
    int obs_dim,
    int lstm_hidden_dim,
    int rollout_depth,
    float gamma,
    bool& done_out,
    EnvState& next_state_out,
    float& total_return_out,
    std::vector<float>* h_out,
    std::vector<float>* c_out,
    TSProfile* prof
) {
    EnvState saved = env.get_state();
    env.set_state(base_state);

    const int n_agents = (int)base_state.cars.size();

    if ((int)obs_seq_flat.size() != seq_len * obs_dim) {
        throw std::runtime_error("controlled obs_seq_flat size mismatch");
    }
    if ((int)h_cur.size() != lstm_hidden_dim) {
        h_cur.assign((size_t)lstm_hidden_dim, 0.0f);
    }
    if ((int)c_cur.size() != lstm_hidden_dim) {
        c_cur.assign((size_t)lstm_hidden_dim, 0.0f);
    }

    // 初始化/重用工作空间
    if (!t_workspace.initialized || (int)t_workspace.other_obs_bufs.size() < n_agents) {
        t_workspace.init(n_agents, seq_len, obs_dim, lstm_hidden_dim);
    }
    t_workspace.reset(n_agents);

    // 使用RingBuffer管理受控智能体的观测序列
    RingBuffer ctrl_obs_buf(seq_len, obs_dim);
    ctrl_obs_buf.from_flat(obs_seq_flat);

    float total_reward = 0.0f;
    float discount = 1.0f;
    bool terminated = false;

    for (int t = 0; t < rollout_depth; ++t) {
        auto all_obs = env.get_observations();

        // 重置动作缓冲区
        std::fill(t_workspace.throttles.begin(), t_workspace.throttles.begin() + n_agents, 0.0f);
        std::fill(t_workspace.steerings.begin(), t_workspace.steerings.begin() + n_agents, 0.0f);

        // 为所有智能体获取均值动作（确定性策略）
        for (int j = 0; j < n_agents; ++j) {
            // t=0时controlled agent跳过，使用传入动作
            if (t == 0 && j == agent_index) continue;
            if (j >= (int)all_obs.size()) continue;

            const std::vector<float>& obsj = all_obs[(size_t)j];
            if ((int)obsj.size() != obs_dim) continue;

            // 懒初始化其他智能体的观测序列
            if (t_workspace.other_obs_bufs[(size_t)j].empty()) {
                t_workspace.other_obs_bufs[(size_t)j].init(seq_len, obs_dim);
                t_workspace.other_obs_bufs[(size_t)j].fill_with(obsj);
            }

            // 获取观测tensor
            t_workspace.other_obs_bufs[(size_t)j].to_flat(t_workspace.temp_obs_flat);
            auto obs_t = make_obs_seq_tensor(t_workspace.temp_obs_flat, seq_len, obs_dim, device);
            auto h_t = make_hc_tensor(t_workspace.other_h[(size_t)j], lstm_hidden_dim, device);
            auto c_t = make_hc_tensor(t_workspace.other_c[(size_t)j], lstm_hidden_dim, device);

            double t0_pv = 0.0;
            if (prof) t0_pv = now_ms();
            auto out_iv = module.run_method("infer_policy_value", obs_t, h_t, c_t);
            if (prof) { prof->ms_infer_pv += (now_ms() - t0_pv); prof->calls_infer_pv += 1; }
            
            auto out_tup = out_iv.toTuple();
            if (!out_tup || out_tup->elements().size() < 3) {
                throw std::runtime_error("infer_policy_value must return (mean,std,value)");
            }

            std::array<float,2> meanj, stdj;
            float vj = 0.0f;
            parse_policy_value_tensors(
                out_tup->elements()[0].toTensor(),
                out_tup->elements()[1].toTensor(),
                out_tup->elements()[2].toTensor(),
                meanj, stdj, vj
            );

            // 确定性：使用均值
            t_workspace.throttles[(size_t)j] = clampf(meanj[0], -1.0f, 1.0f);
            t_workspace.steerings[(size_t)j] = clampf(meanj[1], -1.0f, 1.0f);
        }

        // t>0时，controlled agent也需要推理获取均值
        if (t > 0 && agent_index >= 0 && agent_index < n_agents) {
            ctrl_obs_buf.to_flat(t_workspace.temp_obs_flat);
            auto obs_t = make_obs_seq_tensor(t_workspace.temp_obs_flat, seq_len, obs_dim, device);
            auto h_t = make_hc_tensor(h_cur, lstm_hidden_dim, device);
            auto c_t = make_hc_tensor(c_cur, lstm_hidden_dim, device);

            double t0_pv = 0.0;
            if (prof) t0_pv = now_ms();
            auto out_iv = module.run_method("infer_policy_value", obs_t, h_t, c_t);
            if (prof) { prof->ms_infer_pv += (now_ms() - t0_pv); prof->calls_infer_pv += 1; }
            
            auto out_tup = out_iv.toTuple();
            std::array<float,2> mean_ctrl, std_ctrl;
            float v_ctrl = 0.0f;
            parse_policy_value_tensors(
                out_tup->elements()[0].toTensor(),
                out_tup->elements()[1].toTensor(),
                out_tup->elements()[2].toTensor(),
                mean_ctrl, std_ctrl, v_ctrl
            );

            t_workspace.throttles[(size_t)agent_index] = clampf(mean_ctrl[0], -1.0f, 1.0f);
            t_workspace.steerings[(size_t)agent_index] = clampf(mean_ctrl[1], -1.0f, 1.0f);
        }

        // t=0时使用传入的候选动作
        if (t == 0 && agent_index >= 0 && agent_index < n_agents) {
            t_workspace.throttles[(size_t)agent_index] = agent_action[0];
            t_workspace.steerings[(size_t)agent_index] = agent_action[1];
        }

        // 执行环境步进
        double t0_env = 0.0;
        if (prof) t0_env = now_ms();
        auto res = env.step(t_workspace.throttles, t_workspace.steerings);
        if (prof) { prof->ms_env_step += (now_ms() - t0_env); prof->calls_env_step += 1; }

        if ((int)res.rewards.size() > agent_index && agent_index >= 0) {
            total_reward += discount * res.rewards[(size_t)agent_index];
        }

        // 更新观测序列和LSTM状态
        if (!res.obs.empty()) {
            // 更新其他智能体
            for (int j = 0; j < n_agents && j < (int)res.obs.size(); ++j) {
                if (j == agent_index) continue;
                if (t_workspace.other_obs_bufs[(size_t)j].empty()) continue;

                const std::vector<float>& obsj_next = res.obs[(size_t)j];
                t_workspace.other_obs_bufs[(size_t)j].append(obsj_next);

                // 更新LSTM状态
                float act_j_buf[2] = {t_workspace.throttles[(size_t)j], t_workspace.steerings[(size_t)j]};
                t_workspace.other_obs_bufs[(size_t)j].to_flat(t_workspace.temp_obs_flat);
                auto obs_tj = make_obs_seq_tensor(t_workspace.temp_obs_flat, seq_len, obs_dim, device);
                auto h_tj = make_hc_tensor(t_workspace.other_h[(size_t)j], lstm_hidden_dim, device);
                auto c_tj = make_hc_tensor(t_workspace.other_c[(size_t)j], lstm_hidden_dim, device);
                auto act_tj = torch::from_blob(
                    act_j_buf, {1, 2},
                    torch::TensorOptions().dtype(torch::kFloat32)
                ).clone().to(device);

                double t0_nh = 0.0;
                if (prof) t0_nh = now_ms();
                auto hc_ivj = module.run_method("infer_next_hidden", obs_tj, h_tj, c_tj, act_tj);
                if (prof) { prof->ms_infer_nh += (now_ms() - t0_nh); prof->calls_infer_nh += 1; }
                
                auto hc_tupj = hc_ivj.toTuple();
                if (hc_tupj && hc_tupj->elements().size() >= 2) {
                    tensor_to_vec1d(hc_tupj->elements()[0].toTensor(), t_workspace.other_h[(size_t)j]);
                    tensor_to_vec1d(hc_tupj->elements()[1].toTensor(), t_workspace.other_c[(size_t)j]);
                }
            }

            // 更新受控智能体
            if (agent_index >= 0 && agent_index < (int)res.obs.size()) {
                const std::vector<float>& obs_next = res.obs[(size_t)agent_index];
                ctrl_obs_buf.append(obs_next);

                ctrl_obs_buf.to_flat(t_workspace.temp_obs_flat);
                auto obs_t = make_obs_seq_tensor(t_workspace.temp_obs_flat, seq_len, obs_dim, device);
                auto h_t = make_hc_tensor(h_cur, lstm_hidden_dim, device);
                auto c_t = make_hc_tensor(c_cur, lstm_hidden_dim, device);
                float act_buf[2] = {t_workspace.throttles[(size_t)agent_index], t_workspace.steerings[(size_t)agent_index]};
                auto act_t = torch::from_blob(
                    act_buf, {1, 2},
                    torch::TensorOptions().dtype(torch::kFloat32)
                ).clone().to(device);

                double t0_nh = 0.0;
                if (prof) t0_nh = now_ms();
                auto hc_iv = module.run_method("infer_next_hidden", obs_t, h_t, c_t, act_t);
                if (prof) { prof->ms_infer_nh += (now_ms() - t0_nh); prof->calls_infer_nh += 1; }
                
                auto hc_tup = hc_iv.toTuple();
                if (hc_tup && hc_tup->elements().size() >= 2) {
                    tensor_to_vec1d(hc_tup->elements()[0].toTensor(), h_cur);
                    tensor_to_vec1d(hc_tup->elements()[1].toTensor(), c_cur);
                }
            }
        }

        discount *= gamma;
        if (res.terminated || res.truncated) { terminated = true; break; }
    }

    next_state_out = env.get_state();

    auto obs = env.get_observations();
    std::vector<float> ego_obs;
    if (agent_index >= 0 && agent_index < (int)obs.size()) ego_obs = obs[(size_t)agent_index];

    env.set_state(saved);

    done_out = terminated;
    total_return_out = total_reward;

    if (h_out && c_out) {
        *h_out = std::move(h_cur);
        *c_out = std::move(c_cur);
    }

    return ego_obs;
}

// ============================================================================
// 回溯函数
// ============================================================================

static void backup_lstm(std::vector<std::pair<int,int>>& path, std::vector<LSTMNode>& nodes, float value) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        int nidx = it->first;
        int eidx = it->second;
        nodes[(size_t)nidx].total_N += 1;
        if (eidx >= 0) {
            nodes[(size_t)nidx].edges[(size_t)eidx].N += 1;
            nodes[(size_t)nidx].edges[(size_t)eidx].W += value;
        }
    }
}

static void backup_basic(std::vector<std::pair<int,int>>& path, std::vector<Node>& nodes, float value) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        int nidx = it->first;
        int eidx = it->second;
        nodes[(size_t)nidx].total_N += 1;
        if (eidx >= 0) {
            nodes[(size_t)nidx].edges[(size_t)eidx].N += 1;
            nodes[(size_t)nidx].edges[(size_t)eidx].W += value;
        }
    }
}

// ============================================================================
// 基础MCTS搜索（非LSTM版本）
// ============================================================================

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
) {
    std::mt19937 rng(seed ? seed : std::random_device{}());

    std::vector<Node> nodes;
    nodes.reserve((size_t)num_simulations + 1);

    Node root;
    root.state = root_state;
    nodes.push_back(std::move(root));

    auto expand_node = [&](Node& node, const std::vector<float>& node_obs, bool add_dirichlet) {
        py::list obs_one;
        {
            py::list row;
            for (float v : node_obs) row.append(v);
            obs_one.append(row);
        }

        py::array_t<float> h_arr({1, 1});
        py::array_t<float> c_arr({1, 1});
        h_arr.mutable_at(0,0) = 0.0f;
        c_arr.mutable_at(0,0) = 0.0f;

        py::tuple pv = infer_fn(obs_one, h_arr, c_arr).cast<py::tuple>();
        if (pv.size() < 3) throw std::runtime_error("infer_fn must return (mean,std,value)");

        std::vector<std::array<float,2>> means, stds;
        std::vector<float> values;
        parse_infer_out(pv, means, stds, values);
        node.V = values.empty() ? 0.0f : values[0];

        std::vector<std::array<float,2>> sampled;
        sampled.reserve((size_t)num_action_samples);
        std::vector<float> logps;
        logps.reserve((size_t)num_action_samples);

        for (int k = 0; k < num_action_samples; ++k) {
            float a0 = means[0][0] + stds[0][0] * randn(rng);
            float a1 = means[0][1] + stds[0][1] * randn(rng);
            a0 = clampf(a0, -1.0f, 1.0f);
            a1 = clampf(a1, -1.0f, 1.0f);
            sampled.push_back({a0, a1});
            float lp = gaussian_log_prob(a0, means[0][0], stds[0][0]) + gaussian_log_prob(a1, means[0][1], stds[0][1]);
            logps.push_back(lp);
        }

        softmax_inplace(logps);

        if (add_dirichlet && dirichlet_eps > 0.0f && dirichlet_alpha > 0.0f) {
            std::gamma_distribution<float> gd(dirichlet_alpha, 1.0f);
            std::vector<float> noise((size_t)num_action_samples, 0.0f);
            float s = 0.0f;
            for (int i = 0; i < num_action_samples; ++i) { noise[(size_t)i] = gd(rng); s += noise[(size_t)i]; }
            if (s > 0.0f) {
                for (auto& x : noise) x /= s;
                for (int i = 0; i < num_action_samples; ++i) {
                    logps[(size_t)i] = (1.0f - dirichlet_eps) * logps[(size_t)i] + dirichlet_eps * noise[(size_t)i];
                }
            }
        }

        node.edges.clear();
        node.edges.reserve((size_t)num_action_samples);
        for (int k = 0; k < num_action_samples; ++k) {
            ChildEdge e;
            e.action = sampled[(size_t)k];
            e.prior = logps[(size_t)k];
            node.edges.push_back(std::move(e));
        }
    };

    expand_node(nodes[0], root_obs, true);

    for (int sim = 0; sim < num_simulations; ++sim) {
        std::vector<std::pair<int,int>> path;
        int cur = 0;

        while (true) {
            Node& n = nodes[(size_t)cur];
            if (n.edges.empty()) break;

            float best = -1e9f;
            int best_idx = 0;
            for (int i = 0; i < (int)n.edges.size(); ++i) {
                float s = puct_score_basic(n.total_N, n.edges[(size_t)i].prior, n.edges[(size_t)i].N, n.edges[(size_t)i].W, c_puct);
                if (s > best) { best = s; best_idx = i; }
            }

            path.push_back({cur, best_idx});
            ChildEdge& e = n.edges[(size_t)best_idx];

            if (e.child >= 0) {
                cur = e.child;
                continue;
            }

            // Expand
            bool done = false;
            EnvState next_state;
            float rollout_return = 0.0f;

            EnvState saved = env.get_state();
            env.set_state(n.state);
            auto res = env.step({e.action[0]}, {e.action[1]});
            
            float immediate_r = 0.0f;
            if (!res.rewards.empty()) immediate_r = res.rewards[0];

            next_state = env.get_state();
            done = res.terminated || res.truncated;

            std::vector<float> next_obs;
            if (!res.obs.empty()) next_obs = res.obs[0];

            env.set_state(saved);

            rollout_return = immediate_r;

            Node child;
            child.state = std::move(next_state);

            int child_idx = (int)nodes.size();
            nodes.push_back(std::move(child));
            e.child = child_idx;

            if (!done && !next_obs.empty()) {
                expand_node(nodes[(size_t)child_idx], next_obs, false);
            }

            float leaf_value = nodes[(size_t)child_idx].V;
            float leaf_backup = rollout_return + gamma * leaf_value;
            backup_basic(path, nodes, leaf_backup);
            break;
        }
    }

    const Node& rootn = nodes[0];
    std::vector<float> action_out = {0.0f, 0.0f};

    if (!rootn.edges.empty()) {
        std::vector<float> probs(rootn.edges.size(), 0.0f);
        float sum = 0.0f;
        for (size_t i = 0; i < rootn.edges.size(); ++i) {
            float p = float(std::max(0, rootn.edges[i].N));
            if (temperature > 1e-6f) p = std::pow(p, 1.0f / temperature);
            probs[i] = p;
            sum += p;
        }
        size_t best_i = 0;
        if (sum > 0.0f) {
            for (auto& p : probs) p /= sum;
            std::discrete_distribution<int> dd(probs.begin(), probs.end());
            best_i = (size_t)dd(rng);
        }
        action_out[0] = rootn.edges[best_i].action[0];
        action_out[1] = rootn.edges[best_i].action[1];
    }

    py::dict stats;
    stats["num_simulations"] = num_simulations;
    return {action_out, stats};
}

// ============================================================================
// TorchScript MCTS搜索（使用确定性rollout）
// ============================================================================

std::pair<std::vector<float>, py::dict> mcts_search_lstm_torchscript(
    IntersectionEnv& env,
    const EnvState& root_state,
    const std::vector<float>& root_obs_seq,
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
) {
    TSProfile prof;
    double t_total0 = now_ms();

    std::mt19937 rng(seed ? seed : std::random_device{}());

    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("TorchScript model file not found: " + model_path);
    }

    torch::Device device(torch::kCPU);
    // 使用缓存的模型
    torch::jit::Module& module = get_cached_model(model_path, device);

    std::vector<LSTMNode> nodes;
    nodes.reserve((size_t)num_simulations + 1);

    LSTMNode root;
    root.state = root_state;
    root.h = root_h;
    root.c = root_c;
    root.obs_seq_flat = root_obs_seq;
    nodes.push_back(std::move(root));

    auto expand_node = [&](LSTMNode& node, bool add_dirichlet) {
        auto obs_t = make_obs_seq_tensor(node.obs_seq_flat, seq_len, obs_dim, device);
        auto h_t = make_hc_tensor(node.h, lstm_hidden_dim, device);
        auto c_t = make_hc_tensor(node.c, lstm_hidden_dim, device);

        double t0_pv = 0.0;
        if (&prof) t0_pv = now_ms();
        auto out_iv = module.run_method("infer_policy_value", obs_t, h_t, c_t);
        prof.ms_infer_pv += (now_ms() - t0_pv);
        prof.calls_infer_pv += 1;

        auto out_tup = out_iv.toTuple();
        if (!out_tup || out_tup->elements().size() < 3) {
            throw std::runtime_error("infer_policy_value must return (mean,std,value)");
        }

        std::array<float,2> mean0, std0;
        float value0 = 0.0f;
        parse_policy_value_tensors(
            out_tup->elements()[0].toTensor(),
            out_tup->elements()[1].toTensor(),
            out_tup->elements()[2].toTensor(),
            mean0, std0, value0
        );
        node.V = value0;

        std::vector<std::array<float,2>> sampled;
        sampled.reserve((size_t)num_action_samples);
        std::vector<float> logps;
        logps.reserve((size_t)num_action_samples);

        for (int k = 0; k < num_action_samples; ++k) {
            float a0 = mean0[0] + std0[0] * randn(rng);
            float a1 = mean0[1] + std0[1] * randn(rng);
            a0 = clampf(a0, -1.0f, 1.0f);
            a1 = clampf(a1, -1.0f, 1.0f);
            sampled.push_back({a0, a1});
            float lp = gaussian_log_prob(a0, mean0[0], std0[0]) + gaussian_log_prob(a1, mean0[1], std0[1]);
            logps.push_back(lp);
        }

        softmax_inplace(logps);

        if (add_dirichlet && dirichlet_eps > 0.0f && dirichlet_alpha > 0.0f) {
            std::gamma_distribution<float> gd(dirichlet_alpha, 1.0f);
            std::vector<float> noise((size_t)num_action_samples, 0.0f);
            float s = 0.0f;
            for (int i = 0; i < num_action_samples; ++i) { noise[(size_t)i] = gd(rng); s += noise[(size_t)i]; }
            if (s > 0.0f) {
                for (auto& x : noise) x /= s;
                for (int i = 0; i < num_action_samples; ++i) {
                    logps[(size_t)i] = (1.0f - dirichlet_eps) * logps[(size_t)i] + dirichlet_eps * noise[(size_t)i];
                }
            }
        }

        // 批量获取next_hidden
        std::vector<float> obs_batch_flat;
        obs_batch_flat.reserve((size_t)num_action_samples * (size_t)seq_len * (size_t)obs_dim);
        for (int k = 0; k < num_action_samples; ++k) {
            obs_batch_flat.insert(obs_batch_flat.end(), node.obs_seq_flat.begin(), node.obs_seq_flat.end());
        }

        auto obs_batch_t = torch::from_blob(
            obs_batch_flat.data(),
            {num_action_samples, seq_len, obs_dim},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone().to(device);

        std::vector<float> h_batch((size_t)num_action_samples * (size_t)lstm_hidden_dim, 0.0f);
        std::vector<float> c_batch((size_t)num_action_samples * (size_t)lstm_hidden_dim, 0.0f);
        for (int k = 0; k < num_action_samples; ++k) {
            for (int i = 0; i < lstm_hidden_dim; ++i) {
                h_batch[(size_t)k * (size_t)lstm_hidden_dim + (size_t)i] = 
                    (i < (int)node.h.size()) ? node.h[(size_t)i] : 0.0f;
                c_batch[(size_t)k * (size_t)lstm_hidden_dim + (size_t)i] = 
                    (i < (int)node.c.size()) ? node.c[(size_t)i] : 0.0f;
            }
        }

        // TorchScript wrapper expects h/c as (B,H) or (1,B,H). We pass (1,B,H) here.
        auto h_batch_t = torch::from_blob(
            h_batch.data(),
            {1, num_action_samples, lstm_hidden_dim},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone().to(device);
        auto c_batch_t = torch::from_blob(
            c_batch.data(),
            {1, num_action_samples, lstm_hidden_dim},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone().to(device);

        std::vector<float> action_batch((size_t)num_action_samples * 2);
        for (int k = 0; k < num_action_samples; ++k) {
            action_batch[(size_t)k * 2] = sampled[(size_t)k][0];
            action_batch[(size_t)k * 2 + 1] = sampled[(size_t)k][1];
        }
        auto action_t = torch::from_blob(
            action_batch.data(),
            {num_action_samples, 2},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone().to(device);

        double t0_nh = 0.0;
        t0_nh = now_ms();
        auto hc_iv = module.run_method("infer_next_hidden", obs_batch_t, h_batch_t, c_batch_t, action_t);
        prof.ms_infer_nh += (now_ms() - t0_nh);
        prof.calls_infer_nh += 1;

        auto hc_tup = hc_iv.toTuple();
        if (!hc_tup || hc_tup->elements().size() < 2) {
            throw std::runtime_error("infer_next_hidden must return (h_next,c_next)");
        }

        torch::Tensor h_next_t = hc_tup->elements()[0].toTensor().detach().to(torch::kCPU).contiguous();
        torch::Tensor c_next_t = hc_tup->elements()[1].toTensor().detach().to(torch::kCPU).contiguous();

        node.edges.clear();
        node.edges.reserve((size_t)num_action_samples);
        for (int k = 0; k < num_action_samples; ++k) {
            LSTMEdge e;
            e.action = sampled[(size_t)k];
            e.prior = logps[(size_t)k];
            e.h_next.resize((size_t)lstm_hidden_dim);
            e.c_next.resize((size_t)lstm_hidden_dim);

            const float* h_ptr = h_next_t.data_ptr<float>();
            const float* c_ptr = c_next_t.data_ptr<float>();
            for (int i = 0; i < lstm_hidden_dim; ++i) {
                e.h_next[(size_t)i] = h_ptr[(size_t)k * (size_t)lstm_hidden_dim + (size_t)i];
                e.c_next[(size_t)i] = c_ptr[(size_t)k * (size_t)lstm_hidden_dim + (size_t)i];
            }
            node.edges.push_back(std::move(e));
        }
    };

    expand_node(nodes[0], true);

    // MCTS主循环
    for (int sim = 0; sim < num_simulations; ++sim) {
        std::vector<std::pair<int,int>> path;
        int cur = 0;

        while (true) {
            LSTMNode& n = nodes[(size_t)cur];
            if (n.edges.empty()) break;

            float best = -1e9f;
            int best_idx = 0;
            for (int i = 0; i < (int)n.edges.size(); ++i) {
                float s = puct_score_basic(n.total_N, n.edges[(size_t)i].prior, n.edges[(size_t)i].N, n.edges[(size_t)i].W, c_puct);
                if (s > best) { best = s; best_idx = i; }
            }

            path.push_back({cur, best_idx});
            LSTMEdge& e = n.edges[(size_t)best_idx];

            if (e.child >= 0) {
                cur = e.child;
                continue;
            }

            // 扩展：使用确定性rollout
            bool done = false;
            EnvState next_state;
            float rollout_return = 0.0f;
            std::vector<float> h_after_rollout, c_after_rollout;

            std::vector<float> next_obs = step_env_with_joint_action_torchscript_deterministic(
                env,
                n.state,
                agent_index,
                e.action,
                module,
                device,
                n.obs_seq_flat,
                e.h_next,
                e.c_next,
                seq_len,
                obs_dim,
                lstm_hidden_dim,
                rollout_depth,
                gamma,
                done,
                next_state,
                rollout_return,
                &h_after_rollout,
                &c_after_rollout,
                &prof
            );

            std::vector<float> child_obs_seq = shift_append_obs_seq(n.obs_seq_flat, seq_len, obs_dim, next_obs);

            LSTMNode child;
            child.state = std::move(next_state);
            child.h = std::move(h_after_rollout);
            child.c = std::move(c_after_rollout);
            child.obs_seq_flat = std::move(child_obs_seq);

            int child_idx = (int)nodes.size();
            nodes.push_back(std::move(child));
            e.child = child_idx;

            expand_node(nodes[(size_t)child_idx], false);

            float leaf_value = nodes[(size_t)child_idx].V;
            float leaf_backup = rollout_return + std::pow(gamma, float(rollout_depth)) * leaf_value;
            backup_lstm(path, nodes, leaf_backup);
            break;
        }
    }

    // 根据访问次数选择动作
    const LSTMNode& rootn = nodes[0];
    std::vector<float> action_out = {0.0f, 0.0f};
    int selected_edge = -1;

    if (!rootn.edges.empty()) {
        std::vector<float> probs(rootn.edges.size(), 0.0f);
        float sum = 0.0f;
        for (size_t i = 0; i < rootn.edges.size(); ++i) {
            float p = float(std::max(0, rootn.edges[i].N));
            if (temperature > 1e-6f) p = std::pow(p, 1.0f / temperature);
            probs[i] = p;
            sum += p;
        }
        size_t best_i = 0;
        if (sum > 0.0f) {
            for (auto& p : probs) p /= sum;
            std::discrete_distribution<int> dd(probs.begin(), probs.end());
            best_i = (size_t)dd(rng);
        }
        selected_edge = (int)best_i;
        action_out[0] = rootn.edges[best_i].action[0];
        action_out[1] = rootn.edges[best_i].action[1];
    }

    py::dict stats;
    stats["num_simulations"] = num_simulations;

    if (selected_edge >= 0) {
        py::array_t<float> h_next({lstm_hidden_dim});
        py::array_t<float> c_next({lstm_hidden_dim});
        {
            auto hm = h_next.mutable_unchecked<1>();
            auto cm = c_next.mutable_unchecked<1>();
            for (int i = 0; i < lstm_hidden_dim; ++i) {
                hm(i) = rootn.edges[(size_t)selected_edge].h_next[(size_t)i];
                cm(i) = rootn.edges[(size_t)selected_edge].c_next[(size_t)i];
            }
        }
        stats["h_next"] = h_next;
        stats["c_next"] = c_next;
    }

    stats["lstm_hidden_dim"] = lstm_hidden_dim;

    prof.ms_total = now_ms() - t_total0;
    {
        py::dict p;
        p["ms_total"] = prof.ms_total;
        p["ms_infer_policy_value"] = prof.ms_infer_pv;
        p["ms_infer_next_hidden"] = prof.ms_infer_nh;
        p["ms_env_step"] = prof.ms_env_step;
        p["ms_bookkeeping"] = prof.ms_total - (prof.ms_infer_pv + prof.ms_infer_nh + prof.ms_env_step);
        p["calls_infer_policy_value"] = (unsigned long long)prof.calls_infer_pv;
        p["calls_infer_next_hidden"] = (unsigned long long)prof.calls_infer_nh;
        p["calls_env_step"] = (unsigned long long)prof.calls_env_step;
        stats["profile"] = p;
    }

    return {action_out, stats};
}

// ============================================================================
// LSTM MCTS搜索（Python回调版本，使用确定性rollout）
// ============================================================================

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
) {
    TSProfile prof;
    double t_total0 = now_ms();

    std::mt19937 rng(seed ? seed : std::random_device{}());

    std::vector<LSTMNode> nodes;
    nodes.reserve((size_t)num_simulations + 1);

    LSTMNode root;
    root.state = root_state;
    root.h = root_h;
    root.c = root_c;
    nodes.push_back(std::move(root));

    auto expand_node = [&](LSTMNode& node, const std::vector<float>& node_obs, bool add_dirichlet) {
        py::list obs_one;
        {
            py::list row;
            for (float v : node_obs) row.append(v);
            obs_one.append(row);
        }

        py::array_t<float> h_arr({1, lstm_hidden_dim});
        py::array_t<float> c_arr({1, lstm_hidden_dim});
        {
            auto hm = h_arr.mutable_unchecked<2>();
            auto cm = c_arr.mutable_unchecked<2>();
            for (int i = 0; i < lstm_hidden_dim; ++i) {
                hm(0, i) = (i < (int)node.h.size()) ? node.h[(size_t)i] : 0.0f;
                cm(0, i) = (i < (int)node.c.size()) ? node.c[(size_t)i] : 0.0f;
            }
        }

        double t0_pv = now_ms();
        py::tuple pv = infer_policy_value(obs_one, h_arr, c_arr).cast<py::tuple>();
        prof.ms_infer_pv += (now_ms() - t0_pv);
        prof.calls_infer_pv += 1;
        
        if (pv.size() < 3) throw std::runtime_error("infer_policy_value must return (mean,std,value)");

        std::vector<std::array<float,2>> means, stds;
        std::vector<float> values;
        parse_infer_out(pv, means, stds, values);
        node.V = values.empty() ? 0.0f : values[0];

        std::vector<std::array<float,2>> sampled;
        sampled.reserve((size_t)num_action_samples);
        std::vector<float> logps;
        logps.reserve((size_t)num_action_samples);

        for (int k = 0; k < num_action_samples; ++k) {
            float a0 = means[0][0] + stds[0][0] * randn(rng);
            float a1 = means[0][1] + stds[0][1] * randn(rng);
            a0 = clampf(a0, -1.0f, 1.0f);
            a1 = clampf(a1, -1.0f, 1.0f);
            sampled.push_back({a0, a1});
            float lp = gaussian_log_prob(a0, means[0][0], stds[0][0]) + gaussian_log_prob(a1, means[0][1], stds[0][1]);
            logps.push_back(lp);
        }

        softmax_inplace(logps);

        if (add_dirichlet && dirichlet_eps > 0.0f && dirichlet_alpha > 0.0f) {
            std::gamma_distribution<float> gd(dirichlet_alpha, 1.0f);
            std::vector<float> noise((size_t)num_action_samples, 0.0f);
            float s = 0.0f;
            for (int i = 0; i < num_action_samples; ++i) { noise[(size_t)i] = gd(rng); s += noise[(size_t)i]; }
            if (s > 0.0f) {
                for (auto& x : noise) x /= s;
                for (int i = 0; i < num_action_samples; ++i) {
                    logps[(size_t)i] = (1.0f - dirichlet_eps) * logps[(size_t)i] + dirichlet_eps * noise[(size_t)i];
                }
            }
        }

        // 批量next_hidden
        py::array_t<float> action_arr({num_action_samples, 2});
        {
            auto am = action_arr.mutable_unchecked<2>();
            for (int k = 0; k < num_action_samples; ++k) {
                am(k, 0) = sampled[(size_t)k][0];
                am(k, 1) = sampled[(size_t)k][1];
            }
        }

        py::list obs_batch;
        for (int k = 0; k < num_action_samples; ++k) {
            py::list row;
            for (float v : node_obs) row.append(v);
            obs_batch.append(row);
        }

        py::array_t<float> h_rep({num_action_samples, lstm_hidden_dim});
        py::array_t<float> c_rep({num_action_samples, lstm_hidden_dim});
        {
            auto hm = h_rep.mutable_unchecked<2>();
            auto cm = c_rep.mutable_unchecked<2>();
            for (int k = 0; k < num_action_samples; ++k) {
                for (int i = 0; i < lstm_hidden_dim; ++i) {
                    hm(k, i) = (i < (int)node.h.size()) ? node.h[(size_t)i] : 0.0f;
                    cm(k, i) = (i < (int)node.c.size()) ? node.c[(size_t)i] : 0.0f;
                }
            }
        }

        double t0_nh = now_ms();
        py::tuple hc = infer_next_hidden(obs_batch, h_rep, c_rep, action_arr).cast<py::tuple>();
        prof.ms_infer_nh += (now_ms() - t0_nh);
        prof.calls_infer_nh += 1;
        
        if (hc.size() < 2) throw std::runtime_error("infer_next_hidden must return (h_next,c_next)");

        std::vector<std::vector<float>> h_next_b, c_next_b;
        parse_hc_batch(hc[0], lstm_hidden_dim, h_next_b);
        parse_hc_batch(hc[1], lstm_hidden_dim, c_next_b);

        node.edges.clear();
        node.edges.reserve((size_t)num_action_samples);
        for (int k = 0; k < num_action_samples; ++k) {
            LSTMEdge e;
            e.action = sampled[(size_t)k];
            e.prior = logps[(size_t)k];
            e.h_next = h_next_b[(size_t)k];
            e.c_next = c_next_b[(size_t)k];
            node.edges.push_back(std::move(e));
        }
    };

    expand_node(nodes[0], root_obs, true);

    // MCTS主循环
    for (int sim = 0; sim < num_simulations; ++sim) {
        std::vector<std::pair<int,int>> path;
        int cur = 0;

        while (true) {
            LSTMNode& n = nodes[(size_t)cur];
            if (n.edges.empty()) break;

            float best = -1e9f;
            int best_idx = 0;
            for (int i = 0; i < (int)n.edges.size(); ++i) {
                float s = puct_score_basic(n.total_N, n.edges[(size_t)i].prior, n.edges[(size_t)i].N, n.edges[(size_t)i].W, c_puct);
                if (s > best) { best = s; best_idx = i; }
            }

            path.push_back({cur, best_idx});
            LSTMEdge& e = n.edges[(size_t)best_idx];

            if (e.child >= 0) {
                cur = e.child;
                continue;
            }

            // 扩展：使用确定性rollout
            bool done = false;
            EnvState next_state;
            float rollout_return = 0.0f;
            std::vector<float> h_after_rollout, c_after_rollout;

            std::vector<float> next_obs = step_env_with_joint_action_deterministic(
                env,
                n.state,
                agent_index,
                e.action,
                infer_policy_value,
                &infer_next_hidden,
                &e.h_next,
                &e.c_next,
                lstm_hidden_dim,
                rollout_depth,
                gamma,
                done,
                next_state,
                rollout_return,
                &h_after_rollout,
                &c_after_rollout,
                &prof
            );

            LSTMNode child;
            child.state = std::move(next_state);
            child.h = std::move(h_after_rollout);
            child.c = std::move(c_after_rollout);

            int child_idx = (int)nodes.size();
            nodes.push_back(std::move(child));
            e.child = child_idx;

            if (!done && !next_obs.empty()) {
                expand_node(nodes[(size_t)child_idx], next_obs, false);
            }

            float leaf_value = nodes[(size_t)child_idx].V;
            float leaf_backup = rollout_return + std::pow(gamma, float(rollout_depth)) * leaf_value;
            backup_lstm(path, nodes, leaf_backup);
            break;
        }
    }

    // 选择动作
    const LSTMNode& rootn = nodes[0];
    std::vector<float> action_out = {0.0f, 0.0f};
    int selected_edge = -1;

    if (!rootn.edges.empty()) {
        std::vector<float> probs(rootn.edges.size(), 0.0f);
        float sum = 0.0f;
        for (size_t i = 0; i < rootn.edges.size(); ++i) {
            float p = float(std::max(0, rootn.edges[i].N));
            if (temperature > 1e-6f) p = std::pow(p, 1.0f / temperature);
            probs[i] = p;
            sum += p;
        }
        size_t best_i = 0;
        if (sum > 0.0f) {
            for (auto& p : probs) p /= sum;
            std::discrete_distribution<int> dd(probs.begin(), probs.end());
            best_i = (size_t)dd(rng);
        }
        selected_edge = (int)best_i;
        action_out[0] = rootn.edges[best_i].action[0];
        action_out[1] = rootn.edges[best_i].action[1];
    }

    py::dict stats;
    stats["num_simulations"] = num_simulations;

    if (selected_edge >= 0) {
        py::array_t<float> h_next({lstm_hidden_dim});
        py::array_t<float> c_next({lstm_hidden_dim});
        {
            auto hm = h_next.mutable_unchecked<1>();
            auto cm = c_next.mutable_unchecked<1>();
            for (int i = 0; i < lstm_hidden_dim; ++i) {
                hm(i) = rootn.edges[(size_t)selected_edge].h_next[(size_t)i];
                cm(i) = rootn.edges[(size_t)selected_edge].c_next[(size_t)i];
            }
        }
        stats["h_next"] = h_next;
        stats["c_next"] = c_next;
    }

    stats["lstm_hidden_dim"] = lstm_hidden_dim;

    prof.ms_total = now_ms() - t_total0;
    {
        py::dict p;
        p["ms_total"] = prof.ms_total;
        p["ms_infer_policy_value"] = prof.ms_infer_pv;
        p["ms_infer_next_hidden"] = prof.ms_infer_nh;
        p["ms_env_step"] = prof.ms_env_step;
        p["ms_bookkeeping"] = prof.ms_total - (prof.ms_infer_pv + prof.ms_infer_nh + prof.ms_env_step);
        p["calls_infer_policy_value"] = (unsigned long long)prof.calls_infer_pv;
        p["calls_infer_next_hidden"] = (unsigned long long)prof.calls_infer_nh;
        p["calls_env_step"] = (unsigned long long)prof.calls_env_step;
        stats["profile"] = p;
    }

    return {action_out, stats};
}