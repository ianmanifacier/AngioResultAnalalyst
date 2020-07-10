import sys
import plot_tools_and_loader as ptl
from distutils.util import strtobool
from os import system

def main(argv):
    system("cls")
    #print("length argv : ", len(argv))
    argv0 = argv[0]
    #
    if argv0 == "-h":
        print("\n\n\n")
        print("You summond the help menu from the LAUNCH COMMAND (lcmd.py)")
        print("Don't worry, help is on the way. Hold on tight. \n")
        print("Here is how you can make \na launch command request: \n")
        print("         py lcmd.py -x bool \n")
        print("example: py lcmd.py -r False")
        print(" -r : read from CSV and save as MultiCellTa ")
        print(" -c : calculate MultiCellTA ")
        print(" False : use undetailed labels")
        print(" True : use detailed labels (default) \n\n")
        print("It hopefully helped you out.\n\n")
    elif argv0 == "-c": # read CSV
        detailed = True
        if len(argv)>=2:
            detailed = bool(strtobool(argv[1]))
        print("Reloading", "   detailed =", detailed)
        ptl.calculate_MultiCellTA_storeData(use_detailed_label=detailed)
    elif argv0 == "-r": # read CSV
        detailed = True
        if len(argv)>=2:
            detailed = bool(strtobool(argv[1]))
        print("read CSV", "   detailed =", detailed)
        ptl.read_CSV_storeData(use_detailed_label=detailed)


if __name__ == "__main__":
    main(sys.argv[1:])