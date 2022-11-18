from gfn import GFNAgent

def test_model_only_finds_one_mode():
    agent = GFNAgent(epochs=20)
    agent.sample(10)
    agent.train()
    ave_frac_error, max_frac_error, min_frac_error = agent.compare_env_to_model_policy(
        sample_size=20, plot_filename="./plot_results/one_mode.png"
    )
    # Note l1 errors when modes are not all (or correctly) captured


def test_all_modes():
    agent = GFNAgent(epochs=1000)
    agent.sample(5000)
    agent.train()
    ave_frac_error, max_frac_error, min_frac_error = agent.compare_env_to_model_policy(
        sample_size=5000, plot_filename="./plot_results/all_modes.png"
    )