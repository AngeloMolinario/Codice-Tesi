import pandas as pd
import numpy as np
import random
from collections import defaultdict, Counter
import argparse
import os

def map_age_to_group(age):
    """
    Map individual age values to age group categories.
    
    Age groups:
    - 0: 0-2 years
    - 1: 3-9 years  
    - 2: 10-19 years
    - 3: 20-29 years
    - 4: 30-39 years
    - 5: 40-49 years
    - 6: 50-59 years
    - 7: 60-69 years
    - 8: 70+ years
    
    Args:
        age: individual age value or age category string
        
    Returns:
        age group category (0-8) or -1 for missing values
    """
    if age == -1 or pd.isna(age):  # Missing value
        return -1
    
    # Check if age is already a category (contains "-" or "more")
    if isinstance(age, str):
        age_str = str(age).lower()
        if "-" in age_str or "more" in age_str or "+" in age_str:
            # Age is already categorized, map to appropriate group
            if "0-2" == age_str: return 0
            elif "3-9" == age_str: return 1
            elif "10-19" == age_str: return 2
            elif "20-29" == age_str: return 3
            elif "30-39" == age_str: return 4
            elif "40-49" == age_str: return 5
            elif "50-59" == age_str: return 6
            elif "60-69" == age_str: return 7
            elif "70" in age_str or "more" in age_str or "+" in age_str: return 8
            else: return -1  # Unknown category format
    
    # If age is numeric, perform standard mapping
    try:
        age_num = float(age)
        age_rounded = round(age_num)
        
        if 0 <= age_rounded <= 2: return 0
        elif 3 <= age_rounded <= 9: return 1
        elif 10 <= age_rounded <= 19: return 2
        elif 20 <= age_rounded <= 29: return 3
        elif 30 <= age_rounded <= 39: return 4
        elif 40 <= age_rounded <= 49: return 5
        elif 50 <= age_rounded <= 59: return 6
        elif 60 <= age_rounded <= 69: return 7
        elif age_rounded >= 70: return 8
        else: return -1  # Invalid age
            
    except (ValueError, TypeError):
        return -1  # Cannot convert to numeric

def check_csv(train_csv_path, val_csv_path):
    """
    Checks for duplicate entries in the first column of the train and validation CSV files.

    Args:
        train_csv_path (str): Path to the training CSV file.
        val_csv_path (str): Path to the validation CSV file.
    """
    try:
        print("\n--- Checking for Duplicates ---")
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)

        # Get the first column (path) from each DataFrame
        train_paths = set(train_df.iloc[:, 0])
        val_paths = set(val_df.iloc[:, 0])

        # Find the intersection (duplicates)
        duplicates = train_paths.intersection(val_paths)

        if not duplicates:
            print("Check successful: No duplicate paths found between training and validation sets.")
        else:
            print(f"WARNING: Found {len(duplicates)} duplicate path(s) between the two files!")
            # Print a few examples of duplicates
            for i, path in enumerate(duplicates):
                if i >= 5:
                    print(f"  ... and {len(duplicates) - 5} more.")
                    break
                print(f"  - {path}")
        print(f"{'-'*31}")

    except FileNotFoundError:
        print("Warning: Could not perform duplicate check because one or both CSV files were not found.")
    except Exception as e:
        print(f"An error occurred during the duplicate check: {e}")


def split_csv_file(input_csv, output_dir, train_ratio=0.8, random_seed=42, rename_original_csv=False):
    """
    Esegue uno split stratificato avanzato su un file CSV.
    1. Applica la mappatura dei gruppi di età se la colonna 'Age' è presente.
    2. Garantisce che ogni combinazione di etichette sia rappresentata nel set di validazione.
    3. Riempie il resto del set di validazione casualmente per raggiungere il target ratio.
    4. Salva i file di output nella directory specificata.
    """
    print(f"\n{'='*80}")
    print(f"INTELLIGENT SPLITTING FOR CSV: {os.path.basename(input_csv)}")
    print(f"{'='*80}")

    # --- CHECK PRE-SPLIT ---
    # Se i file di output e il file 'original.csv' esistono già, si assume che lo split sia stato completato.
    # Questo previene riesecuzioni accidentali, specialmente quando si usa 'rename_original_csv'.
    original_csv_in_place = os.path.exists(os.path.join(os.path.dirname(input_csv), 'original.csv'))
    train_labels_exist = os.path.exists(os.path.join(output_dir, 'train', 'labels.csv'))
    val_labels_exist = os.path.exists(os.path.join(output_dir, 'val', 'labels.csv'))

    if original_csv_in_place and train_labels_exist and val_labels_exist:
        print("\nSkipping split: Found 'original.csv', 'train/labels.csv', and 'val/labels.csv'.")
        print("It appears the dataset has already been split.")
        print(f"{'='*80}\n")
        return

    random.seed(random_seed)
    np.random.seed(random_seed)

    try:
        df = pd.read_csv(input_csv, sep=',')
        print(f"Loaded {len(df)} total samples from {input_csv}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Rimuove la colonna dell'indice se è stata salvata per errore nel file di input
    # per garantire che l'header di output corrisponda a quello di input.
    if df.columns[0].startswith('Unnamed:'):
        df = df.iloc[:, 1:]
        print("Removed unnamed index column to match input header format.")

    # Salva una copia della colonna 'Age' originale se esiste e contiene dati
    original_age_col = None
    if 'Age' in df.columns and df['Age'].notna().any():
        print("\nApplying age-to-group mapping on 'Age' column for stratification purposes...")
        original_age_col = df['Age'].copy() # Salva i valori originali
        df['Age'] = df['Age'].apply(map_age_to_group)
        print("Age mapping complete.")

    print("\n--- First 5 rows of the loaded data (after potential age mapping) ---")
    print(df.head().to_string())
    print(f"{'-'*60}\n")

    total_samples = len(df)
    if total_samples == 0:
        print("CSV file is empty. Cannot perform split.")
        return

    if len(df.columns) < 2:
        print(f"Error: CSV must have at least two columns (a path and at least one label).")
        return
    
    # Trova l'indice della colonna del percorso (case-insensitive) e seleziona le colonne successive
    cols = df.columns.tolist()
    path_col_name = 'Path' # Assumiamo che la colonna si chiami 'Path'
    try:
        # Cerca il nome della colonna in modo case-insensitive
        path_col_name_found = next(c for c in cols if c.lower() == path_col_name.lower())
        path_col_index = cols.index(path_col_name_found)
        label_columns = cols[path_col_index + 1:]
    except StopIteration:
        print(f"Warning: Column '{path_col_name}' not found. Defaulting to use all columns after the first one for labels.")
        label_columns = df.columns.tolist()[1:]

    print(f"\nUsing the following columns for stratification: {label_columns}")

    # Step 1: Stratificazione per combinazioni
    combination_indices = defaultdict(list)
    for idx, row in df.iterrows():
        label_combination = tuple(row[col] for col in label_columns if pd.notna(row[col]))
        combination_indices[label_combination].append(idx)
    
    print(f"\nFound {len(combination_indices)} unique label combinations for stratification.")

    # Step 2: Garantire almeno un campione per ogni combinazione nel validation set
    val_indices = set()
    available_indices_per_combo = {combo: list(indices) for combo, indices in combination_indices.items()}

    for combo, indices in available_indices_per_combo.items():
        if len(indices) > 1:
            random.shuffle(indices)
            guaranteed_idx = indices.pop(0)
            val_indices.add(guaranteed_idx)

    print(f"Guaranteed {len(val_indices)} samples in validation set to cover all combinations.")

    # Step 3: Riempimento del validation set fino al ratio desiderato
    target_val_size = round(total_samples * (1 - train_ratio))
    
    all_indices_set = set(df.index)
    remaining_pool = list(all_indices_set - val_indices)
    random.shuffle(remaining_pool)

    num_to_add = target_val_size - len(val_indices)
    if num_to_add > 0:
        additional_indices = remaining_pool[:num_to_add]
        val_indices.update(additional_indices)

    # Step 4: Crea i set finali di indici
    final_val_indices = sorted(list(val_indices))
    final_train_indices = sorted(list(all_indices_set - set(final_val_indices)))

    # Ripristina la colonna 'Age' originale prima di salvare, se è stata modificata
    if original_age_col is not None:
        print("\nRestoring original 'Age' values before saving...")
        df['Age'] = original_age_col

    # Crea i DataFrame di output
    train_df = df.loc[final_train_indices]
    val_df = df.loc[final_val_indices]

    try:
        # Definisci le directory di output e creale
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Definisci i percorsi completi dei file di output
        train_csv_name = 'train.csv'
        validation_csv_name = 'validation.csv'
        if rename_original_csv:
            try:
                original_dir = os.path.dirname(input_csv)
                new_path = os.path.join(original_dir, 'original.csv')
                
                if os.path.abspath(input_csv) == os.path.abspath(new_path):
                    print("\nInfo: Original file is already named 'original.csv'. No need to rename.")
                elif os.path.exists(new_path):
                    print(f"\nWarning: Cannot rename. A file named 'original.csv' already exists in the directory.")
                else:
                    os.rename(input_csv, new_path)
                    print(f"\nSuccessfully renamed original file to: {new_path}")
                train_csv_name = validation_csv_name = 'labels.csv'
            except Exception as e:
                print(f"\nError: Could not rename the original CSV file. Reason: {e}")

            
            
        train_output_path = os.path.join(train_dir, train_csv_name)
        val_output_path = os.path.join(val_dir, validation_csv_name)

        # Salva i file CSV
        print(f"\nSaving training data to: {train_output_path}")
        train_df.to_csv(train_output_path, index=True, sep=',')
        
        print(f"Saving validation data to: {val_output_path}")
        val_df.to_csv(val_output_path, index=True, sep=',')

        print("\n--- Preview of Saved Training Data (first 5 rows) ---")
        print(train_df.head().to_string())
        print("\n--- Preview of Saved Validation Data (first 5 rows) ---")
        print(val_df.head().to_string())

    except Exception as e:
        print(f"\nFATAL ERROR: Could not save the output CSV files.")
        print(f"Reason: {e}")
        return # Interrompe l'esecuzione se il salvataggio fallisce

    # Esegui il controllo per i duplicati
    check_csv(train_output_path, val_output_path)

    # Stampa riepilogo
    final_train_size = len(train_df)
    final_val_size = len(val_df)
    final_train_ratio = final_train_size / total_samples if total_samples > 0 else 0

    print("\n--- FINAL SPLIT RESULTS ---")
    print(f"Target train ratio: {train_ratio:.2%}")
    print(f"Actual train ratio: {final_train_ratio:.2%}")
    print(f"Training set:   {final_train_size} samples")
    print(f"Validation set: {final_val_size} samples")

    # --- Stampa la distribuzione delle classi ---
    print("\n--- Final Class Distribution ---")
    for col in label_columns:
        print(f"\nTask: '{col}'")
        
        train_counts = train_df[col].value_counts()
        val_counts = val_df[col].value_counts()
        
        dist_df = pd.DataFrame({
            'Train Count': train_counts,
            'Val Count': val_counts
        }).fillna(0).astype(int).sort_index()
        
        print(dist_df.to_string())

    print(f"\n\nSuccessfully created:")
    print(f"  - Training data: {train_output_path}")
    print(f"  - Validation data: {val_output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a CSV file into training and validation sets with advanced stratification.")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, default="output_split", help="Directory to save the output CSV files.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion of the dataset for the training set (e.g., 0.8 for 80%).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--rename_original_csv", action='store_true', help="Rename the original CSV file to 'original.csv' before splitting.")
    args = parser.parse_args()
    if 'train' in args.input_csv:
        output_dir = args.input_csv.split('train')[0]
    elif 'test' in args.input_csv:
        output_dir = args.input_csv.split('test')[0]
    else:
        exit(f"Error: Could not determine output directory from input CSV path: {args.input_csv}")
    split_csv_file(
        input_csv=args.input_csv,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
        rename_original_csv=args.rename_original_csv
    )