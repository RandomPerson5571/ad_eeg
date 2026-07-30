"""Backward-compatible wrapper. Use classifier_models.train_mlp instead."""
from classifier_models.train_mlp import train_mlp

train_eeg_model = train_mlp

if __name__ == "__main__":
    from classifier_models.train_mlp import parse_args

    args = parse_args()
    train_mlp(
        test_size=args.test_size,
        hidden_layers=tuple(args.hidden_layers),
        output_path=args.output,
    )
