# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # df = pd.read_csv("results.csv")
# # # df = pd.read_csv("results_train_noWA.csv")
# # df = pd.read_csv("results_withAug.csv")
# # # df = pd.read_csv("results_withWeights.csv")
# # df.columns = df.columns.str.strip()  # Remove whitespace from column names

# # plt.figure()
# # plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
# # plt.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
# # plt.legend()
# # plt.show()

# # plt.figure()
# # plt.plot(df["epoch"], df["train/obj_loss"], label="Train Object Loss")
# # plt.plot(df["epoch"], df["val/obj_loss"], label="Val Object Loss")
# # plt.legend()
# # plt.show()

# # plt.figure()
# # plt.plot(df["epoch"], df["train/cls_loss"], label="Train Class Loss")
# # plt.plot(df["epoch"], df["val/cls_loss"], label="Val Class Loss")
# # plt.legend()
# # plt.show()


# # plt.figure()
# # plt.plot(df["epoch"], df["metrics/mAP_0.5:0.95"], label="mAP@0.5:0.95")
# # plt.plot(df["epoch"], df["metrics/mAP_0.5"], label="mAP@0.5")
# # plt.legend()
# # plt.show()

# # plt.figure()
# # plt.plot(df["epoch"], df["metrics/precision"], label="Precision")
# # plt.plot(df["epoch"], df["metrics/recall"], label="Recall")
# # plt.legend()
# # plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # List of CSV files to process
# csv_files = [
#     "results_without_pretrain_weights.csv",
#     "results_with_pretrain_weights_and_augmentation.csv",
#     "results_with_pretrain_weights.csv"
# ]

# # Ensure output directory exists
# output_dir = "training_plots"
# os.makedirs(output_dir, exist_ok=True)

# for csv_file in csv_files:
#     # Read the CSV file
#     df = pd.read_csv(csv_file)
#     df.columns = df.columns.str.strip()  # Remove whitespace from column names
#     dataset_name = os.path.splitext(csv_file)[0]

#     # Plot 1: Box Loss
#     plt.figure(figsize=(10, 6))
#     plt.plot(df["epoch"], df["train/box_loss"], label="Train Bounding Box Loss")
#     plt.plot(df["epoch"], df["val/box_loss"], label="Val Bounding Box Loss")
#     plt.title(f"Bounding Box Loss vs Epoch - {dataset_name}")
#     plt.xlabel("Epoch")
#     plt.ylabel("Bounding Box Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(output_dir, f"{dataset_name}_box_loss.png"))
#     plt.close()

#     # Plot 2: Object Loss
#     plt.figure(figsize=(10, 6))
#     plt.plot(df["epoch"], df["train/obj_loss"], label="Train Object Loss")
#     plt.plot(df["epoch"], df["val/obj_loss"], label="Val Object Loss")
#     plt.title(f"Object Loss vs Epoch - {dataset_name}")
#     plt.xlabel("Epoch")
#     plt.ylabel("Object Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(output_dir, f"{dataset_name}_obj_loss.png"))
#     plt.close()

#     # Plot 3: Class Loss
#     plt.figure(figsize=(10, 6))
#     plt.plot(df["epoch"], df["train/cls_loss"], label="Train Class Loss")
#     plt.plot(df["epoch"], df["val/cls_loss"], label="Val Class Loss")
#     plt.title(f"Class Loss vs Epoch - {dataset_name}")
#     plt.xlabel("Epoch")
#     plt.ylabel("Class Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(output_dir, f"{dataset_name}_cls_loss.png"))
#     plt.close()

#     # Plot 4: mAP Metrics
#     plt.figure(figsize=(10, 6))
#     plt.plot(df["epoch"], df["metrics/mAP_0.5:0.95"], label="mAP@0.5:0.95")
#     plt.plot(df["epoch"], df["metrics/mAP_0.5"], label="mAP@0.5")
#     plt.title(f"mAP Metrics vs Epoch (Validation) - {dataset_name}")
#     plt.xlabel("Epoch")
#     plt.ylabel("mAP")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(output_dir, f"{dataset_name}_map_metrics.png"))
#     plt.close()

#     # Plot 5: Precision and Recall
#     plt.figure(figsize=(10, 6))
#     plt.plot(df["epoch"], df["metrics/precision"], label="Precision")
#     plt.plot(df["epoch"], df["metrics/recall"], label="Recall")
#     plt.title(f"Precision and Recall vs Epoch - {dataset_name}")
#     plt.xlabel("Epoch")
#     plt.ylabel("Score")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(output_dir, f"{dataset_name}_precision_recall.png"))
#     plt.close()


import os
import pandas as pd
import matplotlib.pyplot as plt

# List of CSV files to process
csv_files = [
    "results-Yolov8.csv"
]

# Ensure output directory exists
output_dir = "training_plots"
os.makedirs(output_dir, exist_ok=True)

for csv_file in csv_files:
    # Read the CSV file
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()  # Remove whitespace from column names
    dataset_name = os.path.splitext(csv_file)[0]

    # Create a single plot for all losses
    plt.figure(figsize=(12, 8))
    
    # Plot Box Loss
    plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", linestyle='-')
    plt.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss", linestyle='--')
    
    # Plot Object Loss
    plt.plot(df["epoch"], df["train/dfl_loss"], label="Train Object Loss", linestyle='-')
    plt.plot(df["epoch"], df["val/dfl_loss"], label="Val Object Loss", linestyle='--')
    
    # Plot Class Loss
    plt.plot(df["epoch"], df["train/cls_loss"], label="Train Class Loss", linestyle='-')
    plt.plot(df["epoch"], df["val/cls_loss"], label="Val Class Loss", linestyle='--')

    # Customize the plot
    plt.title(f"All Losses (Class, Object & Bounding Box) vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_all_losses.png"))
    plt.close()
    
    
        # Plot 4: mAP Metrics
    plt.figure(figsize=(10, 6))
    # plt.plot(df["epoch"], df["metrics/mAP_0.5:0.95"], label="mAP@0.5:0.95")
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5")
    plt.title(f"mAP Metrics vs Epoch (Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_map_metrics.png"))
    plt.close()
