from data import load_or_fetch_historical, compute_features, get_hmm_observations
import numpy as np

for pair in ["BTC/USD", "ETH/USD"]:
    df = load_or_fetch_historical(pair)
    df_features = compute_features(df)
    obs = get_hmm_observations(df_features)
    
    print(f"\n=== {pair} ===")
    print(f"Feature ranges over full training data:")
    print(f"  log_return:    min={obs[:,0].min():.6f}  max={obs[:,0].max():.6f}  mean={obs[:,0].mean():.6f}")
    print(f"  rolling_vol:   min={obs[:,1].min():.6f}  max={obs[:,1].max():.6f}  mean={obs[:,1].mean():.6f}")
    print(f"  momentum:      min={obs[:,2].min():.6f}  max={obs[:,2].max():.6f}  mean={obs[:,2].mean():.6f}")
    print(f"  volume_zscore: min={obs[:,3].min():.6f}  max={obs[:,3].max():.6f}  mean={obs[:,3].mean():.6f}")
    print(f"\nLast 5 rows:")
    print(obs[-5:])
