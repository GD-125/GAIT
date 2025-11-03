"""
Efficient Training Pipeline - Complete Optimized Workflow
Combines all optimizations: parallel data loading, caching, quality filtering,
mixed precision training, and memory optimization.
"""
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import argparse
import time

# Import optimized components
from efficient_data_loader import EfficientGaitDataLoader
from efficient_trainer import EfficientGaitTrainer, create_efficient_data_loaders
from data_quality_checker import DataQualityChecker

# Import existing components
from models import CNN_BiLSTM_GaitDetector, get_device, set_seed, save_training_config
from training import MetricsCalculator, calculate_class_weights
from utils import Config, get_default_config, Visualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Efficient Gait Detection Training')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Directory containing raw CSV files')
    parser.add_argument('--cache_dir', type=str, default='data/cache',
                       help='Directory for caching preprocessed data')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (None = all)')
    parser.add_argument('--skip_quality_check', action='store_true',
                       help='Skip data quality checking')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for optimizer')

    # Optimization arguments
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (use 0 on Windows if issues)')
    parser.add_argument('--max_workers', type=int, default=None,
                       help='Max workers for parallel preprocessing')
    parser.add_argument('--save_memory', action='store_true',
                       help='Enable aggressive memory saving mode')

    # Cache arguments
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='Use caching for preprocessed data')
    parser.add_argument('--clear_cache', action='store_true',
                       help='Clear cache before starting')

    # Model arguments
    parser.add_argument('--window_size', type=int, default=128,
                       help='Window size for segmentation')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap ratio for windows')

    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='results/logs',
                       help='Directory to save logs')
    parser.add_argument('--plots_dir', type=str, default='results/plots',
                       help='Directory to save plots')

    return parser.parse_args()


def main():
    """Main efficient training function."""

    args = parse_args()

    print("\n" + "="*80)
    print("EFFICIENT GAIT DETECTION TRAINING PIPELINE")
    print("="*80 + "\n")

    # Set random seed for reproducibility
    set_seed(42)

    # Create output directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # ==================== STEP 1: Data Quality Check ====================
    if not args.skip_quality_check:
        print("\n[STEP 1/6] Checking Data Quality...")
        checker = DataQualityChecker(args.data_dir)
        quality_report = checker.check_all_files(max_files=args.max_files, save_report=True)

        # Get valid files
        valid_file_paths = checker.get_valid_file_paths()
        print(f"\n✓ Quality check complete: {len(valid_file_paths)} valid files found")

        if not valid_file_paths:
            print("ERROR: No valid files found!")
            return
    else:
        print("\n[STEP 1/6] Skipping quality check...")

    # ==================== STEP 2: Efficient Data Loading ====================
    print("\n[STEP 2/6] Loading and Preprocessing Data (Parallel + Caching)...")

    # Initialize efficient loader
    loader = EfficientGaitDataLoader(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        enable_cache=args.use_cache,
        enable_quality_filter=not args.skip_quality_check,
        max_workers=args.max_workers
    )

    # Clear cache if requested
    if args.clear_cache:
        print("Clearing cache...")
        loader.clear_cache()

    # Pipeline configuration
    pipeline_config = {
        'sampling_rate': 100.0,
        'window_size': args.window_size,
        'overlap': args.overlap,
        'normalization_method': 'zscore',
        'filter_type': 'sensor_specific',
        'balance_method': 'undersample'
    }

    # Load data with parallel processing
    windowed_data, windowed_labels, load_stats = loader.load_files_parallel(
        file_pattern="*.csv",
        max_files=args.max_files,
        pipeline_config=pipeline_config
    )

    # Show cache info
    loader.get_cache_info()

    # ==================== STEP 3: Data Splitting ====================
    print("\n[STEP 3/6] Splitting Data...")

    # Split into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        windowed_data, windowed_labels,
        test_size=0.15,
        random_state=42,
        stratify=windowed_labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.176,  # 0.176 of 0.85 = 0.15 overall
        random_state=42,
        stratify=y_temp
    )

    print(f"Train set: {len(X_train)} samples ({len(X_train)/len(windowed_data)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(windowed_data)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(windowed_data)*100:.1f}%)")

    # ==================== STEP 4: Create Optimized Data Loaders ====================
    print("\n[STEP 4/6] Creating Optimized Data Loaders...")

    train_loader, val_loader = create_efficient_data_loaders(
        X_train, y_train,
        X_val, y_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch_size}")

    # ==================== STEP 5: Model and Training Setup ====================
    print("\n[STEP 5/6] Setting up Model and Training...")

    device = get_device()
    print(f"Training device: {device}")

    # Initialize model
    model = CNN_BiLSTM_GaitDetector(
        input_features=38,
        seq_length=args.window_size,
        conv_filters=[64, 128, 256],
        kernel_sizes=[5, 5, 5],
        lstm_hidden_size=128,
        lstm_num_layers=2,
        fc_hidden_sizes=[256, 128],
        dropout=0.3,
        use_batch_norm=True,
        use_residual=True
    )

    model.print_model_summary()

    # Loss function with class weights (BCEWithLogitsLoss is safe for AMP)
    class_weights = calculate_class_weights(y_train)
    weight_tensor = torch.FloatTensor([class_weights[1] / class_weights[0]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    # Optimizer (AdamW for better generalization)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        #verbose=True
    )

    # Save training configuration
    config = {
        'data': {
            'data_dir': args.data_dir,
            'num_files_processed': load_stats['processed'] + load_stats['cached'],
            'total_samples': len(windowed_data),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
        },
        'model': {
            'input_features': 38,
            'seq_length': args.window_size,
            'architecture': 'CNN-BiLSTM'
        },
        'training': {
            'batch_size': args.batch_size,
            'effective_batch_size': effective_batch_size,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'use_amp': args.use_amp,
            'gradient_accumulation_steps': args.gradient_accumulation_steps
        },
        'optimization': {
            'parallel_workers': args.max_workers,
            'data_workers': args.num_workers,
            'cache_enabled': args.use_cache,
            'quality_filter_enabled': not args.skip_quality_check
        }
    }

    save_training_config(config, f'{args.checkpoint_dir}/training_config.json')

    # ==================== STEP 6: Efficient Training ====================
    print("\n[STEP 6/6] Training Model with All Optimizations...")

    # Create efficient trainer
    trainer = EfficientGaitTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        early_stopping_patience=20,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        save_memory=args.save_memory
    )

    # Train model
    trainer.train(num_epochs=args.epochs, save_best_only=True, log_interval=1)

    # ==================== STEP 7: Evaluation ====================
    print("\n[STEP 7/7] Evaluating on Test Set...")

    # Load best model
    from models import load_checkpoint
    checkpoint = load_checkpoint(
        f'{args.checkpoint_dir}/best_model.pt',
        model,
        device=device
    )

    # Evaluate on test set
    from training import ModelValidator
    validator = ModelValidator(model=model, device=device)
    test_results = validator.test(X_test, y_test, batch_size=args.batch_size)

    # Print test results
    print(f"\n{'='*80}")
    print("TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy:  {test_results['metrics']['accuracy']:.4f}")
    print(f"Precision: {test_results['metrics']['precision']:.4f}")
    print(f"Recall:    {test_results['metrics']['recall']:.4f}")
    print(f"F1-Score:  {test_results['metrics']['f1']:.4f}")
    print(f"AUC-ROC:   {test_results['metrics']['auc_roc']:.4f}")
    print(f"{'='*80}\n")

    # Visualize results
    visualizer = Visualizer(save_dir=args.plots_dir)

    # Plot training history
    visualizer.plot_training_history(
        trainer.history,
        save_path=f'{args.plots_dir}/training_history.png'
    )

    # Plot confusion matrix
    visualizer.plot_confusion_matrix(
        test_results['metrics']['confusion_matrix'],
        save_path=f'{args.plots_dir}/confusion_matrix.png'
    )

    # Plot ROC and PR curves
    from training import MetricsCalculator
    calc = MetricsCalculator()
    roc_data = calc.get_roc_curve_data(
        test_results['labels'],
        test_results['probabilities']
    )
    pr_data = calc.get_pr_curve_data(
        test_results['labels'],
        test_results['probabilities']
    )

    visualizer.plot_roc_curve(
        roc_data['fpr'],
        roc_data['tpr'],
        roc_data['auc'],
        save_path=f'{args.plots_dir}/roc_curve.png'
    )

    visualizer.plot_precision_recall_curve(
        pr_data['precision'],
        pr_data['recall'],
        pr_data['auc_pr'],
        save_path=f'{args.plots_dir}/pr_curve.png'
    )

    # ==================== Completion ====================
    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Total Time: {total_time/60:.2f} minutes")
    print(f"\nResults saved to:")
    print(f"  - Checkpoints: {args.checkpoint_dir}")
    print(f"  - Logs: {args.log_dir}")
    print(f"  - Plots: {args.plots_dir}")
    print(f"\nOptimization Summary:")
    print(f"  - Files processed: {load_stats['processed']}")
    print(f"  - Files from cache: {load_stats['cached']}")
    print(f"  - Files skipped: {load_stats['skipped']}")
    print(f"  - Mixed precision: {'Enabled' if args.use_amp else 'Disabled'}")
    print(f"  - Effective batch size: {effective_batch_size}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
