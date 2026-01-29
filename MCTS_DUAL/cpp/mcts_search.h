#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <vector>
#include <utility>

class IntersectionEnv;
struct EnvState;

namespace py = pybind11;

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
