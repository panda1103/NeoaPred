# Copyright 2023 AIDuhl Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

from NeoaPred.predicter import neoantigen_predicter

def main(args):
    for s in args.step:
        try:
            assert(s in ["0", "1", "2", "3"])
        except:
            print("Step Parameter Error.\nExample: \"--step 0 1 2 3\" ")

    neoantigen_predicter(
                            args.input_file,
                            args.output_dir,
                            args.step,
                            args.trained_model_1,
                            args.trained_model_2,
                        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input file (*.csv)",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getcwd(),
        help="Output directory (default = ./)",
        required=False,
    )
    parser.add_argument(
        '--step',
        nargs='*',
        default=["0","1","2","3"],
        help="Step (default = 0 1 2 3)\n\n\
0: Construct initialized linear peptide.\n\
1: Generate HLA-I peptide complex structure.\n\
2: Compute molecular surfaces features.\n\
3: Predict foreignness score.\n\n",
        required=False,
    )
    parser.add_argument(
        '--trained_model_1',
        default=os.path.dirname(os.path.abspath(__file__))+"/NeoaPred/PepConf/trained_model/model_1.pth",
        help="Pre-trained model for step 1.\n(default = NeoaPred/PepConf/trained_model/model_1.pth)",
        required=False,
    )
    parser.add_argument(
        '--trained_model_2',
        default=os.path.dirname(os.path.abspath(__file__))+"/NeoaPred/PepFore/trained_model/model_2.pth",
        help="Pre-trained model for step 4.\n(default = NeoaPred/PepFore/trained_model/model_2.pth)",
        required=False,
    )
    args = parser.parse_args()
    main(args)
