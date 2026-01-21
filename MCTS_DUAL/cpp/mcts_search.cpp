#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <random>
#include <vector>
#include <utility>
#include <algorithm>
#include <limits>

#include "IntersectionEnv.h"
#include "EnvState.h"

namespace py = pybind11;

struct ChildEdge {
    std::array<float, 2> action; // throttle, steer
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

static inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

static inline float randn(std::mt19937& rng) {
    static thread_local std::normal_distribution<float> nd(0.0f, 1.0f);
    return nd(rng);
}

// Helper to compute log prob of a sample from a univariate Gaussian
static inline float gaussian_log_prob(float x, float mean, float std) {
    if (std <= 1e-6f) return -std::numeric_limits<float>::infinity();
    const float var = std * std;
    const float log_std = std::log(std);
    constexpr float log_2pi = 1.83787706641f; // log(2*PI)
    return -0.5f * ((x - mean) * (x - mean) / var + log_2pi) - log_std;
}

// Helper to compute softmax in-place with numerical stability
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

static py::tuple call_infer(const py::function& infer_fn, const std::vector<std::vector<float>>& obs_batch) {
    py::list lst;
    lst.attr("reserve")(obs_batch.size());
    for (const auto& obs : obs_batch) {
        py::list row;
        row.attr("reserve")(obs.size());
        for (float v : obs) row.append(v);
        lst.append(row);
    }
    return infer_fn(lst).cast<py::tuple>();
}

static void parse_infer_out(const py::tuple& out,
                            std::vector<std::array<float,2>>& means,
                            std::vector<std::array<float,2>>& stds,
                            std::vector<float>& values) {
    // out = (mean, std, value). Each can be list-like or numpy.
    py::array mean_arr = py::array::ensure(out[0]);
    py::array std_arr  = py::array::ensure(out[1]);
    py::array val_arr  = py::array::ensure(out[2]);

    // 修正：为 unchecked<> 提供数据类型
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

static void expand_node(IntersectionEnv& env,
                        Node& node,
                        const std::vector<float>& node_obs,
                        const py::function& infer_fn,
                        int num_action_samples,
                        float dirichlet_alpha,
                        float dirichlet_eps,
                        std::mt19937& rng) {
    // Get mean/std/value from network
    py::tuple out = infer_fn(py::cast(std::vector<std::vector<float>>{node_obs})).cast<py::tuple>();

    std::vector<std::array<float,2>> means, stds;
    std::vector<float> values;
    parse_infer_out(out, means, stds, values);

    node.V = values.empty() ? 0.0f : values[0];

    // Sample K actions from Gaussian, compute prior from Gaussian log-prob
    node.edges.clear();
    node.edges.reserve((size_t)num_action_samples);

    std::vector<float> logps;
    logps.reserve((size_t)num_action_samples);

    for (int k = 0; k < num_action_samples; ++k) {
        float a0 = means[0][0] + stds[0][0] * randn(rng);
        float a1 = means[0][1] + stds[0][1] * randn(rng);
        a0 = clampf(a0, -1.0f, 1.0f);
        a1 = clampf(a1, -1.0f, 1.0f);

        ChildEdge e;
        e.action = {a0, a1};
        node.edges.push_back(e);

        float lp = gaussian_log_prob(a0, means[0][0], stds[0][0]) + gaussian_log_prob(a1, means[0][1], stds[0][1]);
        logps.push_back(lp);
    }

    // Convert log-probs to normalized priors
    softmax_inplace(logps);

    // Optional Dirichlet noise (should be applied only at root; callers pass eps/alpha accordingly)
    if (dirichlet_eps > 0.0f && dirichlet_alpha > 0.0f) {
        std::gamma_distribution<float> gd(dirichlet_alpha, 1.0f);
        std::vector<float> noise((size_t)num_action_samples, 0.0f);
        float sum = 0.0f;
        for (int i = 0; i < num_action_samples; ++i) { noise[(size_t)i] = gd(rng); sum += noise[(size_t)i]; }
        if (sum > 0.0f) {
            for (auto& x : noise) x /= sum;
            for (int i = 0; i < num_action_samples; ++i) {
                logps[(size_t)i] = (1.0f - dirichlet_eps) * logps[(size_t)i] + dirichlet_eps * noise[(size_t)i];
            }
        }
    }

    for (int k = 0; k < num_action_samples; ++k) {
        node.edges[(size_t)k].prior = logps[(size_t)k];
    }
}

static float puct_score(const Node& node, const ChildEdge& e, float c_puct) {
    const float Q = (e.N > 0) ? (e.W / float(e.N)) : 0.0f;
    const float U = c_puct * e.prior * std::sqrt(float(std::max(1, node.total_N))) / (1.0f + float(e.N));
    return Q + U;
}

static std::vector<float> step_env_with_action(IntersectionEnv& env,
                                               const EnvState& base_state,
                                               const std::array<float,2>& action,
                                               int rollout_depth,
                                               float gamma,
                                               bool& done_out,
                                               EnvState& next_state_out) {
    // Save current env
    EnvState saved = env.get_state();
    env.set_state(base_state);

    float total_reward = 0.0f;
    float discount = 1.0f;
    bool terminated = false;

    // Only ego actions (single ego mode expected for traffic_flow). If multi-agent, we apply to all.
    for (int t = 0; t < rollout_depth; ++t) {
        auto res = env.step({action[0]}, {action[1]});
        if (!res.rewards.empty()) total_reward += discount * res.rewards[0];
        discount *= gamma;
        if (res.terminated || res.truncated) { terminated = true; break; }
    }

    next_state_out = env.get_state();

    // Get obs for ego
    auto obs = env.get_observations();
    std::vector<float> ego_obs;
    if (!obs.empty()) ego_obs = obs[0];

    // Restore
    env.set_state(saved);

    done_out = terminated;
    return ego_obs;
}

static void backup(std::vector<std::pair<int,int>>& path, std::vector<Node>& nodes, float value) {
    // path elements: (node_idx, edge_idx) for chosen edge in that node; last node has edge_idx=-1
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        int nidx = it->first;
        int eidx = it->second;
        nodes[nidx].total_N += 1;
        if (eidx >= 0) {
            nodes[nidx].edges[(size_t)eidx].N += 1;
            nodes[nidx].edges[(size_t)eidx].W += value;
        }
        // Optional discount by depth can be applied; keep as-is for now.
    }
}

std::pair<std::vector<float>, py::dict> mcts_search(
    IntersectionEnv& env,
    const EnvState& root_state,
    const std::vector<float>& root_obs,
    const py::function& infer_fn,
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

    // Expand root once
    expand_node(env, nodes[0], root_obs, infer_fn, num_action_samples, dirichlet_alpha, dirichlet_eps, rng);

    for (int sim = 0; sim < num_simulations; ++sim) {
        std::vector<std::pair<int,int>> path; // (node_idx, edge_idx)
        int cur = 0;

        // Selection
        while (true) {
            Node& n = nodes[(size_t)cur];
            if (n.edges.empty()) {
                break;
            }
            // pick best edge
            float best = -1e9f;
            int best_e = 0;
            for (int ei = 0; ei < (int)n.edges.size(); ++ei) {
                float s = puct_score(n, n.edges[(size_t)ei], c_puct);
                if (s > best) { best = s; best_e = ei; }
            }

            path.push_back({cur, best_e});

            ChildEdge& e = n.edges[(size_t)best_e];
            if (e.child < 0) {
                // Expand new child by stepping env
                bool done = false;
                EnvState next_state;
                auto next_obs = step_env_with_action(env, n.state, e.action, rollout_depth, gamma, done, next_state);

                Node child;
                child.state = std::move(next_state);
                int child_idx = (int)nodes.size();
                nodes.push_back(std::move(child));
                e.child = child_idx;

                // Expand child (network eval) and get value
                expand_node(env, nodes[(size_t)child_idx], next_obs, infer_fn, num_action_samples, 0.0f, 0.0f, rng);
                float leaf_value = nodes[(size_t)child_idx].V;

                // If terminal, override value with 0 (or could use rollout reward already included in V logic).
                // Keep leaf_value.
                backup(path, nodes, leaf_value);
                break;
            } else {
                cur = e.child;
                // continue down
            }
        }
    }

    // Choose action from root visits
    const Node& rootn = nodes[0];
    std::vector<float> action_out = {0.0f, 0.0f};

    if (!rootn.edges.empty()) {
        // compute probabilities from N
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
            // sample
            std::discrete_distribution<int> dd(probs.begin(), probs.end());
            best_i = (size_t)dd(rng);
        } else {
            // fallback
            best_i = 0;
        }

        action_out[0] = rootn.edges[best_i].action[0];
        action_out[1] = rootn.edges[best_i].action[1];
    }

    py::dict stats;
    py::dict visit_counts;
    for (size_t i = 0; i < rootn.edges.size(); ++i) {
        py::list a;
        a.append(rootn.edges[i].action[0]);
        a.append(rootn.edges[i].action[1]);
        visit_counts[py::str(py::repr(a))] = rootn.edges[i].N;
    }
    stats["visit_counts"] = visit_counts;
    stats["num_simulations"] = num_simulations;

    return {action_out, stats};
}

