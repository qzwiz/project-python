
import pickle
with open("features.pkl", "rb") as f:
    feature_cols = pickle.load(f)

    print(len(feature_cols))
    
    print(feature_cols[:10])

