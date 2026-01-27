#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
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

struct LSTMEdge {
    std::array<float, 2> action; // throttle, steer
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
};

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
    constexpr float log_2pi = 1.83787706641f; // log(2*PI)
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
        // (B,H)
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
        // (B,1,H)
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

static std::vector<float> step_env_with_action(IntersectionEnv& env,
                                               const EnvState& base_state,
                                               const std::array<float,2>& action,
                                               int rollout_depth,
                                               float gamma,
                                               bool& done_out,
                                               EnvState& next_state_out) {
    EnvState saved = env.get_state();
    env.set_state(base_state);

    float total_reward = 0.0f;
    float discount = 1.0f;
    bool terminated = false;

    for (int t = 0; t < rollout_depth; ++t) {
        auto res = env.step({action[0]}, {action[1]});
        if (!res.rewards.empty()) total_reward += discount * res.rewards[0];
        discount *= gamma;
        if (res.terminated || res.truncated) { terminated = true; break; }
    }

    next_state_out = env.get_state();

    auto obs = env.get_observations();
    std::vector<float> ego_obs;
    if (!obs.empty()) ego_obs = obs[0];

    env.set_state(saved);

    done_out = terminated;
    (void)total_reward;
    return ego_obs;
}

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

    auto expand_node = [&](Node& node, const std::vector<float>& node_obs, bool add_dirichlet) {
        // We reuse the same infer_policy_value signature as LSTM version: (obs_batch, h, c) -> (mean,std,value)
        // For non-LSTM, we pass dummy h/c.
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

        // Sample actions + priors
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
            node.edges.push_back(e);
        }
    };

    // Expand root
    expand_node(nodes[0], root_obs, true);

    // MCTS loop
    for (int sim = 0; sim < num_simulations; ++sim) {
        std::vector<std::pair<int,int>> path;
        int cur = 0;

        while (true) {
            Node& n = nodes[(size_t)cur];
            if (n.edges.empty()) break;

            float best = -1e9f;
            int best_e = 0;
            for (int ei = 0; ei < (int)n.edges.size(); ++ei) {
                const ChildEdge& e = n.edges[(size_t)ei];
                float s = puct_score_basic(n.total_N, e.prior, e.N, e.W, c_puct);
                if (s > best) { best = s; best_e = ei; }
            }

            path.push_back({cur, best_e});

            ChildEdge& e = n.edges[(size_t)best_e];
            if (e.child < 0) {
                bool done = false;
                EnvState next_state;
                auto next_obs = step_env_with_action(env, n.state, e.action, rollout_depth, gamma, done, next_state);

                Node child;
                child.state = std::move(next_state);

                int child_idx = (int)nodes.size();
                nodes.push_back(std::move(child));
                e.child = child_idx;

                expand_node(nodes[(size_t)child_idx], next_obs, false);

                float leaf_value = nodes[(size_t)child_idx].V;
                backup_basic(path, nodes, leaf_value);
                break;
            }

            cur = e.child;
        }
    }

    // Choose action from root visits
    const Node& rootn = nodes[0];
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

std::pair<std::vector<float>, py::dict> mcts_search_lstm(
    IntersectionEnv& env,
    const EnvState& root_state,
    const std::vector<float>& root_obs,
    const py::function& infer_policy_value,
    const py::function& infer_next_hidden,
    const std::vector<float>& root_h,
    const std::vector<float>& root_c,
    int lstm_hidden_dim,
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

    std::vector<LSTMNode> nodes;
    nodes.reserve((size_t)num_simulations + 1);

    LSTMNode root;
    root.state = root_state;
    root.h = root_h;
    root.c = root_c;
    nodes.push_back(std::move(root));

    auto expand_node = [&](LSTMNode& node, const std::vector<float>& node_obs, bool add_dirichlet) {
        // policy/value call with batch size 1
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

        py::tuple pv = infer_policy_value(obs_one, h_arr, c_arr).cast<py::tuple>();
        if (pv.size() < 3) throw std::runtime_error("infer_policy_value must return (mean,std,value)");

        std::vector<std::array<float,2>> means, stds;
        std::vector<float> values;
        parse_infer_out(pv, means, stds, values);
        node.V = values.empty() ? 0.0f : values[0];

        // Sample actions + priors
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

        // Batch next_hidden for each sampled action
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

        py::tuple hc = infer_next_hidden(obs_batch, h_rep, c_rep, action_arr).cast<py::tuple>();
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

    // Expand root
    expand_node(nodes[0], root_obs, true);

    // MCTS loop
    for (int sim = 0; sim < num_simulations; ++sim) {
        std::vector<std::pair<int,int>> path;
        int cur = 0;

        while (true) {
            LSTMNode& n = nodes[(size_t)cur];
            if (n.edges.empty()) break;

            float best = -1e9f;
            int best_e = 0;
            for (int ei = 0; ei < (int)n.edges.size(); ++ei) {
                const LSTMEdge& e = n.edges[(size_t)ei];
                float s = puct_score_basic(n.total_N, e.prior, e.N, e.W, c_puct);
                if (s > best) { best = s; best_e = ei; }
            }

            path.push_back({cur, best_e});

            LSTMEdge& e = n.edges[(size_t)best_e];
            if (e.child < 0) {
                bool done = false;
                EnvState next_state;
                auto next_obs = step_env_with_action(env, n.state, e.action, rollout_depth, gamma, done, next_state);

                LSTMNode child;
                child.state = std::move(next_state);
                child.h = e.h_next;
                child.c = e.c_next;

                int child_idx = (int)nodes.size();
                nodes.push_back(std::move(child));
                e.child = child_idx;

                expand_node(nodes[(size_t)child_idx], next_obs, false);

                float leaf_value = nodes[(size_t)child_idx].V;
                backup_lstm(path, nodes, leaf_value);
                break;
            }

            cur = e.child;
        }
    }

    // Choose action from root visits
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
    py::dict visit_counts;
    for (size_t i = 0; i < rootn.edges.size(); ++i) {
        py::list a;
        a.append(rootn.edges[i].action[0]);
        a.append(rootn.edges[i].action[1]);
        visit_counts[py::str(py::repr(a))] = rootn.edges[i].N;
    }
    stats["visit_counts"] = visit_counts;
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

    return {action_out, stats};
}
