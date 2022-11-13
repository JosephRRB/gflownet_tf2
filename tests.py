from gfn import GFNAgent

def test():
    agent = GFNAgent(epochs=200)
    agent.sample(10)
    agent.train()
    agent

