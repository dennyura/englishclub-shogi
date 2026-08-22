"""Generate Full Shogi Minimax-supervised data for download from Kaggle."""

from __future__ import annotations

import argparse
from pathlib import Path

from shogi_ai.game.full_shogi.state import FullShogiState
from shogi_ai.training.minimax_dataset import generate_minimax_data, save_training_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--opening-moves", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=100.0)
    parser.add_argument("--max-moves", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("minimax_full_shogi.pt"))
    args = parser.parse_args()

    examples = generate_minimax_data(
        initial_state=FullShogiState(),
        num_games=args.games,
        depth=args.depth,
        opening_moves=args.opening_moves,
        top_k=args.top_k,
        temperature=args.temperature,
        max_moves=args.max_moves,
        num_workers=args.workers,
    )
    save_training_examples(examples, args.output)
    print(f"saved {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
