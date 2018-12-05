"""
This script contains some useful commandline combinations of the package
argparse
"""
import argparse

parser = argparse.ArgumentParser(description="Hello. I am the parser. Please provide "
                                             "some arguments for this program.")

# Add some optional arguments
parser.add_argument('--output_folder',
                    type=str,
                    help="Specify the folder to put the calculated data.")
