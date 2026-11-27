import argparse
from datetime import datetime, timezone

from daneel.detection import TransitModel
from daneel.parameters import Parameters
from daneel.detection.CNN_class import CNNclassifier_class
from daneel.detection.RF_class import randomforest
from daneel.dream.light_curve_gan import GAN_class

def main():
    parser = argparse.ArgumentParser(description="Model and analyse exoplanet transits.")

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input par file to pass",
    )

    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("-t", "--transit", action="store_true", help="Plot a transit light curve")

    actions.add_argument("-d", "--detect", type=str, choices=["rf", "cnn"], help="Run exoplanet detection")
    actions.add_argument("-a", "--atmosphere", action="store_true", help="Run atmospheric characterisation")
    actions.add_argument("--dream", action="store_true", help="Run GAN to generate lightcurves")
    parser.add_argument("-o", "--output", default="lc.png", help="Output image for --transit")

    args = parser.parse_args()

    """Launch Daneel"""
    start = datetime.now(timezone.utc)

    input_pars = Parameters(args.input_file).params

    if args.transit:
        transit = TransitModel(input_pars['transit'])
        transit.plot_light_curve(args.output)
    elif args.detect == "rf":
        prediction = randomforest(input_pars['RandomForest'])
        prediction.run_RandomForest()
    elif args.detect == "cnn":
        prediction = CNNclassifier_class(input_pars['CNN'])
        prediction.run_CNN() 
    elif arg.dream:
        gan = GAN_class(input_pars['GAN'])
        gan_lc = run_GAN()
        gan.plot_lightcurves(gan_lc)

    elif args.atmosphere:
        parser.error("atmospheric characterisation is not implemented yet; use --transit")

    finish = datetime.now(timezone.utc)
    print(f"Daneel completed in {(finish - start).total_seconds():.2f}s")


if __name__ == "__main__":
    main()
