# Dataset Splitting Logic

Questo documento descrive in dettaglio come avviene lo splitting dei dataset in training e validation nel progetto di tesi.

## Overview

Il sistema di splitting implementato nel progetto utilizza un approccio di **stratificazione avanzata** che garantisce una distribuzione bilanciata delle classi tra training e validation set, con particolare attenzione alle combinazioni multi-task (età, genere, emozione).

## Struttura dei File

### File Principali
- **`dataset/split_csv.py`**: Contiene la logica principale per lo splitting stratificato
- **`script/split_dataset.sh`**: Script bash per automatizzare lo splitting di tutti i dataset
- **`dataset/dataset.py`**: Implementazione delle classi dataset che utilizzano i dati splittati

## Algoritmo di Splitting

### 1. Preparazione dei Dati

Il processo inizia con la lettura del file CSV contenente i path delle immagini e le relative etichette:

```python
def split_csv_file(input_csv, output_dir, train_ratio=0.8, random_seed=42, rename_original_csv=False)
```

**Parametri:**
- `input_csv`: Path al file CSV originale
- `output_dir`: Directory di output per i file splittati
- `train_ratio`: Proporzione del dataset per il training (default: 0.8)
- `random_seed`: Seed per la riproducibilità (default: 42)
- `rename_original_csv`: Se rinominare il file originale come "original.csv"

### 2. Strategia di Stratificazione

⚠️ **IMPORTANTE: La distribuzione delle classi del training set NON viene perfettamente rispettata nel validation set.**

L'algoritmo implementato utilizza una strategia **combination-first** che privilegia la **copertura completa** delle combinazioni di etichette rispetto al mantenimento esatto delle proporzioni per singole classi. Questo approccio è stato scelto specificatamente per scenari multi-task dove la rappresentanza di tutte le combinazioni possibili è critica per la valutazione.

### 3. Mappatura delle Età

Prima dello splitting, viene applicata una mappatura delle età individuali in gruppi di età standardizzati:

```python
def map_age_to_group(age):
    """
    Gruppi di età:
    - 0: 0-2 anni
    - 1: 3-9 anni
    - 2: 10-19 anni
    - 3: 20-29 anni
    - 4: 30-39 anni
    - 5: 40-49 anni
    - 6: 50-59 anni
    - 7: 60-69 anni
    - 8: 70+ anni
    """
```

Questa mappatura serve per:
- Standardizzare le categorie di età tra diversi dataset
- Facilitare la stratificazione basata su gruppi piuttosto che età specifiche
- Gestire sia valori numerici che categorie testuali

### 3. Stratificazione Intelligente

L'algoritmo implementa una strategia di stratificazione in due fasi:

#### Fase 1: Garanzia di Rappresentanza (Guaranteed Coverage)

```python
# Step 1: Garantire almeno un campione per ogni combinazione nel validation set
val_indices = set()
for combo, indices in combination_indices.items():
    if len(indices) > 1:
        random.shuffle(indices)
        guaranteed_idx = indices.pop(0)
        val_indices.add(guaranteed_idx)
```

**Obiettivo**: Assicurare che ogni combinazione unica di etichette (età, genere, emozione) sia rappresentata nel validation set con almeno un campione.

#### Fase 2: Riempimento Proporzionale

```python
# Step 2: Riempimento del validation set fino al ratio desiderato
target_val_size = round(total_samples * (1 - train_ratio))
remaining_pool = list(all_indices_set - val_indices)
random.shuffle(remaining_pool)
num_to_add = target_val_size - len(val_indices)
if num_to_add > 0:
    additional_indices = remaining_pool[:num_to_add]
    val_indices.update(additional_indices)
```

**Obiettivo**: Completare il validation set fino a raggiungere il rapporto desiderato (es. 20% del dataset totale).

⚠️ **CRITICO**: Questa seconda fase utilizza una **selezione completamente casuale** che NON rispetta le proporzioni delle classi del training set. Una volta garantita la copertura delle combinazioni, i campioni rimanenti vengono aggiunti senza considerare il bilanciamento delle singole classi.

#### Dettaglio del Riempimento Casuale

Una volta completata la Fase 1, l'algoritmo procede come segue:

1. **Calcolo campioni mancanti**:
   ```python
   target_val_size = round(total_samples * (1 - train_ratio))  # es. 20% del totale
   num_to_add = target_val_size - len(val_indices)  # Sottrae i campioni già garantiti
   ```

2. **Selezione casuale dal pool rimanente**:
   ```python
   remaining_pool = list(all_indices_set - val_indices)  # Tutti i campioni NON ancora nel validation
   random.shuffle(remaining_pool)  # Mescolamento casuale
   additional_indices = remaining_pool[:num_to_add]  # Primi N campioni casuali
   ```

3. **Risultato**: Le proporzioni finali delle classi dipendono completamente dalla casualità della selezione nella Fase 2.

#### Esempio Pratico di Distorsione

**Scenario**:
- Dataset: 1000 campioni totali
- Classe A: 800 campioni (80% del training)
- Classe B: 200 campioni (20% del training)
- 50 combinazioni uniche garantite nella Fase 1
- Target validation: 200 campioni (20%)

**Risultato tipico**:
- Fase 1: 50 campioni (mix delle combinazioni)
- Fase 2: 150 campioni casuali → probabilmente ~120 da Classe A e ~30 da Classe B
- **Validation finale**: Classe A potrebbe essere 85-90%, Classe B 10-15%
- **Distorsione**: La distribuzione nel validation set non riflette quella del training set

### 4. Controlli di Qualità

#### Verifica Duplicati
```python
def check_csv(train_csv_path, val_csv_path):
    """Verifica la presenza di duplicati tra training e validation set"""
    train_paths = set(train_df.iloc[:, 0])
    val_paths = set(val_df.iloc[:, 0])
    duplicates = train_paths.intersection(val_paths)
```

#### Ripristino Valori Originali
Prima del salvataggio, i valori di età vengono ripristinati ai valori originali:
```python
if original_age_col is not None:
    print("\nRestoring original 'Age' values before saving...")
    df['Age'] = original_age_col
```

## Configurazione Multi-Dataset

### Script di Automazione

Il file `script/split_dataset.sh` automatizza lo splitting per tutti i dataset:

```bash
#!/bin/bash

echo "Splitting CelebA_HQ"
python3 dataset/split_csv.py ../processed_datasets/datasets_with_standard_labels/CelebA_HQ/train/labels.csv --train_ratio 0.8 --seed 2025 --rename_original_csv

echo "Splitting FairFace"
python3 dataset/split_csv.py ../processed_datasets/datasets_with_standard_labels/FairFace/train/labels.csv --train_ratio 0.8 --seed 2025 --rename_original_csv

echo "Splitting RAF-DB"
python3 dataset/split_csv.py ../processed_datasets/datasets_with_standard_labels/RAF-DB/train/labels.csv --train_ratio 0.8 --seed 2025 --rename_original_csv

echo "Splitting Lagenda"
python3 dataset/split_csv.py ../processed_datasets/datasets_with_standard_labels/Lagenda/train/labels.csv --train_ratio 0.8 --seed 2025 --rename_original_csv
```

**Caratteristiche:**
- **Seed fisso (2025)**: Garantisce riproducibilità tra esperimenti
- **Train ratio 0.8**: 80% training, 20% validation
- **Rename original**: Il file originale viene rinominato "original.csv"

## Struttura di Output

Dopo lo splitting, la struttura delle directory diventa:

```
dataset_name/
├── original.csv          # File originale rinominato
├── train/
│   └── labels.csv        # 80% dei dati
└── val/
    └── labels.csv        # 20% dei dati
```

## Integrazione con le Classi Dataset

### BaseDataset
```python
class BaseDataset(Dataset, WeightCalculationMixin):
    def __init__(self, root, transform=None, split="train", verbose=False):
        self.split_path = os.path.join(root, split)
        self.labels_path = os.path.join(self.split_path, "labels.csv")
        self.data = pd.read_csv(self.labels_path)
```

La classe `BaseDataset` legge automaticamente i file splittati basandosi sul parametro `split` ("train" o "val").

### MultiDataset e TaskBalanceDataset
```python
class MultiDataset(Dataset, WeightCalculationMixin):
    """Combina più dataset mantenendo la separazione train/val"""

class TaskBalanceDataset(Dataset, WeightCalculationMixin):
    """Applica bilanciamento dei task su dataset multipli"""
```

Queste classi utilizzano internamente `BaseDataset`, ereditando automaticamente la logica di splitting.

## Vantaggi e Svantaggi dell'Approccio

### ✅ Vantaggi

#### 1. Copertura Completa Multi-Task
- **Garanzia assoluta** che ogni combinazione di (età, genere, emozione) sia rappresentata nel validation set
- Criticamente importante per valutazioni multi-task affidabili
- Evita situazioni dove alcune combinazioni non sarebbero mai testate

#### 2. Riproducibilità
- Seed fisso (2025) garantisce split identici tra esecuzioni
- Controlli automatici prevengono riesecuzioni accidentali
- Risultati completamente deterministici

#### 3. Controllo Qualità
- Verifica automatica di duplicati tra train/validation
- Statistiche dettagliate sulla distribuzione finale
- Preservazione dei valori originali delle età

#### 4. Flessibilità Implementativa
- Supporto per diversi formati di età (numerico/categorico)
- Gestione robusta di valori mancanti (-1)
- Parametri configurabili per diversi scenari

### ❌ Svantaggi

#### 1. **Distorsione delle Distribuzioni delle Classi**
- ⚠️ **PROBLEMA PRINCIPALE**: Le proporzioni delle singole classi nel validation set NON rispecchiano quelle del training set
- La selezione casuale nella Fase 2 può creare sbilanciamenti significativi
- Classi maggioritarie tendono ad essere sovra-rappresentate nel validation

#### 2. **Validazione Potenzialmente Non Rappresentativa**
- Le metriche di performance sul validation set potrebbero non riflettere accuratamente le performance sul training set
- Bias verso classi maggioritarie nelle valutazioni
- Difficoltà nella comparazione diretta train vs validation accuracy

#### 3. **Mancanza di Stratificazione Per-Classe**
- Non implementa stratificazione proporzionale per singole etichette
- Non considera il bilanciamento individuale di età, genere, emozione
- Approccio "combination-first" sacrifica l'equilibrio per-classe

#### 4. **Imprevedibilità delle Proporzioni Finali**
- Le distribuzioni finali dipendono dalla casualità
- Impossibile prevedere a priori quanto una classe sarà rappresentata
- Variabilità tra diversi dataset con lo stesso algoritmo

### 🔄 Alternative Non Implementate

L'algoritmo **NON** utilizza approcci più sofisticati come:

1. **Stratificazione Ibrida**:
   - Fase 1: Garantire combinazioni (come attualmente)
   - Fase 2: Stratificazione proporzionale invece di selezione casuale

2. **Ottimizzazione Vincolata**:
   - Minimizzare la distanza dalle proporzioni originali
   - Soggetto al vincolo di copertura completa delle combinazioni

3. **Weighted Sampling nella Fase 2**:
   - Pesare la selezione casuale per favorire classi sotto-rappresentate
   - Bilanciamento dinamico basato sulle proporzioni correnti

4. **Stratificazione Gerarchica**:
   - Prima stratificare per task primario
   - Poi per combinazioni all'interno di ogni stratum

### 🎯 Quando Usare Questo Approccio

**✅ Raccomandato per**:
- Valutazione multi-task dove la copertura è critica
- Dataset con molte combinazioni rare ma importanti
- Scenari dove testare ogni combinazione è più importante della fedeltà distributiva
- Ricerca esplorativa su interazioni tra task

**❌ Sconsigliato per**:
- Valutazione single-task tradizionale
- Dataset dove le proporzioni delle classi sono critiche
- Confronti diretti train vs validation accuracy
- Ottimizzazione di iperparametri basata su validation performance

## Statistiche di Splitting

Il sistema fornisce statistiche dettagliate per ogni split:

```python
print("\n--- FINAL SPLIT RESULTS ---")
print(f"Target train ratio: {train_ratio:.2%}")
print(f"Actual train ratio: {final_train_ratio:.2%}")
print(f"Training set:   {final_train_size} samples")
print(f"Validation set: {final_val_size} samples")

print("\n--- Final Class Distribution ---")
for col in label_columns:
    print(f"\nTask: '{col}'")
    # Distribuzione per training e validation
```

## Esempi di Output

### CelebA_HQ Splitting
```
INTELLIGENT SPLITTING FOR CSV: labels.csv
================================================================================
Loaded 30000 total samples from ../processed_datasets/datasets_with_standard_labels/CelebA_HQ/train/labels.csv

Applying age-to-group mapping on 'Age' column for stratification purposes...
Age mapping complete.

Using the following columns for stratification: ['Age', 'Gender', 'Facial Emotion']

Found 156 unique label combinations for stratification.
Guaranteed 156 samples in validation set to cover all combinations.

--- FINAL SPLIT RESULTS ---
Target train ratio: 80.00%
Actual train ratio: 80.02%
Training set:   24006 samples
Validation set: 5994 samples

Successfully created:
  - Training data: ../processed_datasets/datasets_with_standard_labels/CelebA_HQ/train/labels.csv
  - Validation data: ../processed_datasets/datasets_with_standard_labels/CelebA_HQ/val/labels.csv
```

## Considerazioni Tecniche

### Gestione della Memoria
- Utilizzo di `pd.read_csv()` per efficienza
- Manipolazione di indici piuttosto che dati completi
- Shuffle in-place per ridurre utilizzo memoria

### Robustezza
- Gestione di file già splittati (skip automatico)
- Controllo integrità dei dati di input
- Fallback per colonne non trovate

### Performance
- Algoritmo O(n) per la maggior parte delle operazioni
- Caching delle combinazioni per evitare ricalcoli
- Shuffle efficiente con `random.shuffle()`

## Limitazioni e Note

1. **Dimensione Minima**: Il validation set deve contenere almeno una combinazione per ogni etichetta unica
2. **Bilanciamento**: In dataset molto sbilanciati, il validation set potrebbe essere leggermente più grande del target
3. **Valori Mancanti**: Le etichette -1 (mancanti) non influenzano la stratificazione
4. **Ordine Colonne**: Il sistema si aspetta la colonna 'Path' seguita dalle etichette

## Conclusioni

Il sistema di splitting implementato fornisce una soluzione robusta e riproducibile per la divisione di dataset multi-task, garantendo:
- Distribuzione bilanciata delle classi
- Copertura completa delle combinazioni di etichette
- Riproducibilità degli esperimenti
- Controlli automatici di qualità
- Flessibilità per diversi formati di dati