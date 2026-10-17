import pickle
import sys
import gymnasium as gym
import numpy as np
from stable_worldmodel.policy import BasePolicy


def _install_legacy_numpy_pickle_aliases() -> None:
    """Expose NumPy private module aliases used by older SB3 checkpoints."""
    try:
        import numpy.core as np_core

        sys.modules.setdefault('numpy._core', np_core)
        sys.modules.setdefault('numpy._core.numeric', np_core.numeric)
        sys.modules.setdefault('numpy._core.multiarray', np_core.multiarray)
        sys.modules.setdefault(
            'numpy._core._multiarray_umath',
            np_core._multiarray_umath,
        )
    except Exception:
        pass


def _load_pickle_with_legacy_rng_compat(path: str):
    """Load older SB3 pickle files that embed legacy NumPy RNG objects."""
    _install_legacy_numpy_pickle_aliases()

    try:
        import numpy.random._pickle as np_pickle
    except Exception:
        with open(path, 'rb') as f:
            return pickle.load(f)

    original_bit_generator_ctor = getattr(
        np_pickle, '__bit_generator_ctor', None
    )
    original_generator_ctor = getattr(np_pickle, '__generator_ctor', None)

    class _DummyBitGenerator:
        def __setstate__(self, state):
            self.state = state

    class _DummyGenerator:
        def __init__(self, bit_generator=None):
            self.bit_generator = bit_generator

        def __setstate__(self, state):
            self.state = state

    try:
        if original_bit_generator_ctor is not None:
            setattr(
                np_pickle,
                '__bit_generator_ctor',
                lambda bit_generator_name='MT19937': _DummyBitGenerator(),
            )
        if original_generator_ctor is not None:
            setattr(
                np_pickle,
                '__generator_ctor',
                lambda bit_generator_name='MT19937', bit_generator_ctor=None: _DummyGenerator(),
            )

        with open(path, 'rb') as f:
            return pickle.load(f)
    finally:
        if original_bit_generator_ctor is not None:
            setattr(
                np_pickle,
                '__bit_generator_ctor',
                original_bit_generator_ctor,
            )
        if original_generator_ctor is not None:
            setattr(
                np_pickle,
                '__generator_ctor',
                original_generator_ctor,
            )


class SB3Normalizer:
    """Normalizer for observations using Stable Baselines3 VecNormalize statistics."""

    def __init__(
        self,
        stats_path: str,
        clip_obs: float = 10.0,
        epsilon: float = 1e-8,
    ):
        vec_normalize = _load_pickle_with_legacy_rng_compat(stats_path)

        self.mean = vec_normalize.obs_rms.mean
        self.var = vec_normalize.obs_rms.var
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.normalize(obs)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        norm_obs = (obs - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(norm_obs, -self.clip_obs, self.clip_obs)


class ExpertPolicy(BasePolicy):
    def __init__(
        self,
        ckpt_path: str,
        vec_normalize_path: str,
        noise_std: float = 0.0,
        seed: int = None,
        device: str = 'cpu',
        **kwargs,
    ):
        """
        Expert Policy (RL) for DMControl Suite.
        Policies have been trained using Soft Actor-Critic (SAC) algorithm from stable_baselines3.

        Args:
            ckpt_path (str): Path to the stable_baselines3 .zip file of the trained policy.
            vec_normalize_path (str): Path to the .pkl file containing the normalization statistics.
            noise_std (float): Standard deviation of Gaussian noise added to actions.
            seed (int): Random seed for action noise.
            device (str): Device to load the model on, e.g., 'cpu' or 'cuda'.
        """
        super().__init__(**kwargs)

        try:
            import stable_baselines3 as sb3
        except ImportError:
            raise ImportError(
                'stable_baselines3 is required to use the ExpertPolicy. '
                "Please install it via 'uv add stable-baselines3'."
            )
        self._sb3 = sb3
        self._ckpt_path = str(ckpt_path)
        self._device = device

        _install_legacy_numpy_pickle_aliases()

        self.normalizer = SB3Normalizer(vec_normalize_path)
        self.model = None
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std
        self.type = 'expert'

    def _load_model(self, env) -> None:
        obs_space = getattr(env, 'single_observation_space', None)
        if not isinstance(obs_space, gym.Space):
            obs_space = env.observation_space

        action_space = getattr(env, 'single_action_space', None)
        if not isinstance(action_space, gym.Space):
            action_space = env.action_space

        self.model = self._sb3.SAC.load(
            self._ckpt_path,
            device=self._device,
            custom_objects={
                'observation_space': obs_space,
                'action_space': action_space,
            },
        )
        if self.seed is not None:
            self.model.set_random_seed(self.seed)

    def set_env(self, env):
        self.env = env
        if self.model is None:
            self._load_model(env)

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if self.model is not None:
            self.model.set_random_seed(seed)

    def get_action(self, info_dict, **kwargs):
        assert 'observation' in info_dict, (
            'Observation key missing in info_dict'
        )

        obs = info_dict['observation'].squeeze()

        if obs.ndim == 1:
            obs = obs[None, :]

        if len(obs.shape) != 2:
            raise ValueError(
                f'Expected observation shape (num_envs, obs_dim), got {obs.shape}'
            )

        normalized_obs = self.normalizer(obs)
        actions, _ = self.model.predict(normalized_obs, deterministic=False)
        if self.noise_std > 0:
            noise = self.rng.normal(0, self.noise_std, size=actions.shape)
            actions = actions + noise
        return np.clip(actions, -1.0, 1.0).astype(np.float32)
