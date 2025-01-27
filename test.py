from gfn import GFNAgent, _plot_l1_errors_per_probability_interval

def test_model_only_finds_one_mode():
    agent = GFNAgent(epochs=1000)
    agent.sample(20)
    agent.train()
    agent_prob, env_prob = agent.compare_env_to_model_policy(
        sample_size=5000, plot_filename="plot_results/one_mode.png"
    )
    # Note l1 errors when modes are not all (or correctly) captured
    errors_per_interval = _plot_l1_errors_per_probability_interval(
        agent_prob, env_prob, "one_mode_errors.png", n_intervals=4
    )

def test_all_modes():
    agent = GFNAgent(epochs=1000)
    agent.sample(5000)
    agent.train()
    agent_prob, env_prob = agent.compare_env_to_model_policy(
        sample_size=5000, plot_filename="plot_results/all_modes.png"
    )
    errors_per_interval = _plot_l1_errors_per_probability_interval(
        agent_prob, env_prob, "all_mode_errors.png", n_intervals=4
    )

def test_grid_5d():
    agent = GFNAgent(env_dim=5, env_len=8, epochs=1000)
    agent.sample(20)
    agent.train()
    agent_prob, env_prob = agent.compare_env_to_model_policy(
        sample_size=50, plot_filename=None
    )
    errors_per_interval = _plot_l1_errors_per_probability_interval(
        agent_prob, env_prob, "grid_5d.png", n_intervals=4
    )