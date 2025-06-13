import optuna

from train_rl_agent_optuna import train


def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.90, 0.99)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    distance_coeff = trial.suggest_float("distance_coeff", 5.0, 20.0)
    obstacle_penalty = trial.suggest_float("obstacle_penalty", -20.0, -1.0)

    avg_reward = train(
        episodes=10,
        steps_per_episode=30,
        trial=trial,
        custom_hparams={
            "lr": lr,
            "gamma": gamma,
            "epsilon_decay": epsilon_decay,
            "hidden_size": hidden_size,
            "distance_coeff": distance_coeff,
            "obstacle_penalty": obstacle_penalty,
        }
    )
    return avg_reward


def run_optuna():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best trial:")
    print(study.best_trial)

if __name__ == "__main__":
    run_optuna()
