# CS294-112 HW 1: Imitation Learning

### Python version:  **3.6.6**

### Steps

1. Install dependencies on your local environment: `pip install -r requirements.txt`
2. Run `sh collect_experts_demo.sh` to collect the demonstrations of the expert
3. Run `python3.6 behavioural_cloning.py`
   - if you already done this step once, then you don't need to train the model again, so just run `python3.6 behavioural_cloning.py --test`
4. Run `python3.6 DAgger.py`
   -  if you already done this step once, then you don't need to train the model again, so just run `python3.6 DAgger.py --test`

### Directory Structure

- `expert_data`: where we store the experts' demo data in Numpy Array format
- `expert_models`: we store the experts' models (DQN and Duelling DQN)
- `videos`: we store the experts' demo video while we collect the demo by `collect_experts_demo.sh`
- `weights`: we store the trained models' weights, e.g., Behavioural Cloning and DAgger