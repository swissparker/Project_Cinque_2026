import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import torch
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from dataset import generate_synthetic_student_data, load_ednet_kt1, preprocess_data
from model import AtRiskModel
from federated import average_weights, train_local_model, evaluate_model
from explainability import explain_with_shap, explain_with_lime

def main():
    parser = argparse.ArgumentParser(description="Explainable Federated Learning for At-Risk Student Detection")
    parser.add_argument("--data", type=str, default="synthetic", choices=["synthetic", "ednet"],
                        help="Which dataset to use (default: synthetic)")
    parser.add_argument("--ednet-path", type=str, default="data/EdNet-KT1.zip",
                        help="Path to EdNet-KT1.zip file (default: data/EdNet-KT1.zip)")
    parser.add_argument("--num-students", type=int, default=10000,
                        help="Number of students to use from EdNet (default: 10000)")
    parser.add_argument("--num-institutions", type=int, default=3,
                        help="Number of federated institutions (default: 3)")
    parser.add_argument("--num-rounds", type=int, default=10,
                        help="Number of federated training rounds (default: 10)")
    parser.add_argument("--local-epochs", type=int, default=5,
                        help="Number of local training epochs per round (default: 5)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    
    args = parser.parse_args()
    
    if args.data == "ednet":
        print(f"Loading EdNet dataset from {args.ednet_path}...")
        if not os.path.exists(args.ednet_path):
            print(f"Error: EdNet file not found at {args.ednet_path}")
            print("Please download EdNet-KT1 from https://github.com/riiid/ednet and place it in the data/ directory")
            print("Using synthetic data instead...")
            raw_data = generate_synthetic_student_data(
                num_students_per_institution=args.num_students // args.num_institutions,
                num_institutions=args.num_institutions
            )
        else:
            raw_data = load_ednet_kt1(
                zip_path=args.ednet_path,
                num_students=args.num_students,
                num_institutions=args.num_institutions
            )
    else:
        print("Generating synthetic student data...")
        raw_data = generate_synthetic_student_data(
            num_students_per_institution=args.num_students // args.num_institutions,
            num_institutions=args.num_institutions
        )
    
    processed_data, scaler, feature_cols = preprocess_data(raw_data)
    input_dim = len(feature_cols)
    print(f"Number of features: {input_dim}")
    
    # Initialize global model
    global_model = AtRiskModel(input_dim=input_dim)
    
    print(f"Starting federated training for {args.num_rounds} rounds...")
    
    for round_num in range(args.num_rounds):
        local_weights = []
        
        # Train each institution's local model
        for inst_id, (X_inst, y_inst) in enumerate(processed_data):
            local_model = AtRiskModel(input_dim=input_dim)
            local_model.load_state_dict(global_model.state_dict())
            
            # Train local model
            trained_weights = train_local_model(
                local_model, X_inst, y_inst,
                epochs=args.local_epochs, lr=args.lr
            )
            local_weights.append(trained_weights)
        
        # Average weights (FedAvg)
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        
        # Evaluate on all institutions
        accuracies = []
        for (X_inst, y_inst) in processed_data:
            acc = evaluate_model(global_model, X_inst, y_inst)
            accuracies.append(acc)
        
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"Round {round_num + 1}/{args.num_rounds}: Average accuracy = {avg_accuracy:.4f}")
    
    print("\nFederated training complete!")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(global_model.state_dict(), "models/trained_model.pth")
    print("Model saved to models/trained_model.pth")
    
    # Explain predictions with SHAP and LIME
    print("\nGenerating explanations...")
    X_test, y_test = processed_data[0]
    shap_values = explain_with_shap(global_model, X_test, feature_cols, sample_idx=0)
    lime_exp = explain_with_lime(global_model, X_test, feature_cols, sample_idx=0)
    
    print("\nSHAP explanation generated!")
    print("LIME explanation generated!")
    
    os.makedirs("results", exist_ok=True)
    fig = lime_exp.as_pyplot_figure()
    fig.savefig("results/lime_explanation.png")
    print("Explanations saved to results/")


if __name__ == "__main__":
    main()