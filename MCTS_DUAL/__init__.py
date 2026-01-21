# MCTS Multi-Agent Training Package

try:
    from .networks import DualNetwork
    from .mcts import MCTS
    from .train import MCTSTrainer
except ImportError:
    from networks import DualNetwork
    from mcts import MCTS
    from train import MCTSTrainer

__all__ = ['MCTS', 'DualNetwork', 'MCTSTrainer']
