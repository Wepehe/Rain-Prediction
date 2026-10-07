"""One-time entry point for the frozen Cycle-2 FINAL evaluation."""
from __future__ import annotations
import argparse,json
from ontario_nowcast.training.cycle2_final_evaluation import generate_final,score_final,write_hash_manifest

def main():
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=["generate","score","hash"]);args=parser.parse_args()
    result=generate_final() if args.phase=="generate" else score_final() if args.phase=="score" else write_hash_manifest()
    print(json.dumps(result,indent=2,allow_nan=True))

if __name__=="__main__":main()
