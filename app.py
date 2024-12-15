# Step 1: Load and Filter Dataset
def load_and_filter_dataset(dataset):
    print("Loading and filtering dataset...")
    filtered_dataset = dataset[~dataset['weight_class'].str.contains("Women", case=False, na=False)]
    filtered_dataset = filtered_dataset[filtered_dataset['Winner.1'] != "Draw"]
    if 'REDFLAG' in filtered_dataset.columns:
        filtered_dataset = filtered_dataset[filtered_dataset['REDFLAG'] == True]
    print(f"Dataset filtered to {filtered_dataset.shape[0]} rows and {filtered_dataset.shape[1]} columns.")
    return filtered_dataset


# Prepare Task Data with Debugging
def prepare_task_data(filtered_dataset, target_column):
    print(f"Preparing data for {target_column} prediction...")
    if target_column not in filtered_dataset.columns:
        print(f"Target column '{target_column}' not found in dataset. Skipping this task.")
        return None, None, None, None

    try:
        X, y = preprocess_dataset(filtered_dataset, target_column)
        X = feature_selection(X, y)
        print(f"Data prepared for {target_column}.")
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except Exception as e:
        print(f"Error preparing data for {target_column}: {e}")
        return None, None, None, None

# Main Execution
print("Starting Updated UFC Fight Prediction Pipeline...")

try:
    fight_data_path = '/content/fighter_vs_fighter_stats.csv'
    main_data_path = '/content/full_dataset_with_all_metrics.csv'

    # Load datasets
    fight_df = pd.read_csv(fight_data_path)
    main_df = pd.read_csv(main_data_path)

    # Filter and preprocess datasets
    filtered_main_df = load_and_filter_dataset(main_df)
    filtered_main_df = compute_differential_metrics(filtered_main_df)

    # Win/Loss Prediction
    task_1_data = prepare_task_data(filtered_main_df, 'Winner.1')
    if task_1_data[0] is not None:
        print("Starting Win/Loss Prediction...")
        X_train_win_loss, X_test_win_loss, y_train_win_loss, y_test_win_loss = task_1_data
        win_loss_model, win_loss_pred, win_loss_proba = train_and_evaluate(
            X_train_win_loss, y_train_win_loss, X_test_win_loss, y_test_win_loss, "Win/Loss Prediction"
        )
    else:
        print("No data available for Win/Loss prediction.")

    # Method of Victory Prediction
    task_2_data = prepare_task_data(filtered_main_df, 'method_of_victory')
    if task_2_data[0] is not None:
        print("Starting Method of Victory Prediction...")
        X_train_mov, X_test_mov, y_train_mov, y_test_mov = task_2_data
        mov_model, mov_pred, mov_proba = train_and_evaluate(
            X_train_mov, y_train_mov, X_test_mov, y_test_mov, "Method of Victory Prediction"
        )
    else:
        print("No data available for Method of Victory prediction.")

    # Round Prediction
    task_3_data = prepare_task_data(filtered_main_df, 'round')
    if task_3_data[0] is not None:
        print("Starting Round Prediction...")
        X_train_round, X_test_round, y_train_round, y_test_round = task_3_data
        round_model, round_pred, round_proba = train_and_evaluate(
            X_train_round, y_train_round, X_test_round, y_test_round, "Round Prediction"
        )
    else:
        print("No data available for Round prediction.")

    print("Pipeline completed successfully.")

except Exception as e:
    print(f"An error occurred in the pipeline: {e}")
