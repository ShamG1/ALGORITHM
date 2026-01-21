#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <vector>
#include <utility>

// Forward declarations to keep header light
class IntersectionEnv;
struct EnvState;

namespace py = pybind11;

// The main MCTS search function to be exposed to Python
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
);
