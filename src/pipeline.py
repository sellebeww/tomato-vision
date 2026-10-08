"""One command: verified dataset -> scratch CNN search -> honest evaluation."""
import argparse
import subprocess
import sys
from src.config import ROOT_DIR
from src.manifest import prepare

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output", required=True, help="New experiment directory")
    p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--samples-per-class",type=int,default=300)
    p.add_argument("--with-synthetic",action="store_true",help="add AI-generated images (tagged synthetic_ai) to train only")
    a=p.parse_args()
    dataset=prepare("own_plus_synthetic" if a.with_synthetic else "own")
    subprocess.run([sys.executable,"-m","src.experiment","--dataset",str(dataset),
                    "--output",a.output,"--epochs",str(a.epochs),
                    "--samples-per-class",str(a.samples_per_class)],cwd=ROOT_DIR,check=True)

if __name__=="__main__":
    main()
