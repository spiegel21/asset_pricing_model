from preprocessing import preprocess
from capm import fit_capm
from model import train_model
from eval import evaluate_models

def main():
    """Main function to preprocess the raw stock data"""
    preprocess()
    fit_capm()
    train_model()
    evaluate_models()

    return None

if __name__ == "__main__":
    main()